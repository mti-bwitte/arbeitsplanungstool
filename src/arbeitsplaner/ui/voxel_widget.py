from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLAxisItem
from pyqtgraph.Vector import Vector
import trimesh

from core.cadservice import Component

class VoxelWidget(QWidget):
    
    """
    Interactive OpenGL viewer for a voxelised 'trimesh'-mesh model.

    The widget renders the binary voxel grid:
    Filled voxel-cell -> True
    Empty voxel-cell -> False
    
    Then constructs a big voxel-trimesh with this 'voxgrid':
    trimesh with n cubes * 12 faces

    Finally 'GLMeshItem' uses 'GLViewWidget' to visualise the 'cube_mesh'
    by extracting its faces and vertices.

    This visualisation gets embedded inside a border-less 'QVBoxLayout'. 
    It can be added to any Qt Layout or used standalone as a floating window.

    Parameter
    ---------
    voxgrid : trimesh.voxel.VoxelGrid
        Pre-computed voxel grid to be visualised.
    parent : QWidget, optional
        Qt parent widget for normal Qt memory management.
        'None' means the widget is a top-level window when shown.
    camera_distance : float or None, optional
        If given, the camera starts at this distance.  
        Pass ``None`` (default) to place the camera at twice the
        bounding‑box diagonal **automatically**.
    surface_only : bool, default False
        ``True`` → render only the outer surface (Marching‑Cubes),
        ``False`` → render every filled voxel as a cube.
    """

    # Colour per component (R, G, B, A)
    _COLOR = {
        Component.WORKPIECE: (0.8, 0.1, 0.1, 1.0),  # red
        Component.FIXTURE:   (0.2, 0.5, 0.8, 1.0),  # blue-ish
        Component.TOOL:      (0.1, 0.8, 0.1, 1.0),  # green
    }

    # Emits the filled-voxel count each time the mesh is (re)built
    voxelCountChanged = pyqtSignal(int)

    def __init__(
        self,
        voxgrid,
        parent=None,
        *,
        camera_distance: float | None = None,
        surface_only: bool = False,
    ):
        
        """
        Construct an empty OpenGL viewer; items are added later via
        update_component().
        """

        super().__init__(parent)

        # Layout & OpenGL-View
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.gl_view = gl.GLViewWidget()
        layout.addWidget(self.gl_view)
        self.gl_view.setBackgroundColor("w")

        # Holds the GLMeshItem for each component slot
        self._items: dict[Component, gl.GLMeshItem] = {}
        self._axis: GLAxisItem | None = None

        # Keep per‑component bounding boxes: {comp: (min_vec, max_vec)}
        self._bounds_dict: dict[Component, tuple[np.ndarray, np.ndarray]] = {}

        # Remember the current global scene size to avoid unnecessary re‑scaling
        self._global_ext: np.ndarray | None = None

        # Viewer starts empty; items are added via update_component(...)
        self.default_distance = 150.0
        self.gl_view.setCameraPosition(distance=self.default_distance)

    # -------------------------------------------------------------
    def _build_item(
        self,
        component: Component,
        *,
        voxgrid=None,
        mesh=None,
        surface_only: bool = True,
    ):
        """
        Add or replace a 3‑D item in the viewer.

        Parameters
        ----------
        component : Component
            One of WORKPIECE, FIXTURE, TOOL.
        voxgrid : trimesh.voxel.VoxelGrid, optional
            Binary voxel model to visualise (workpiece / fixture).
        mesh : trimesh.Trimesh, optional
            Trimesh to display (tool or raw fixture).
        surface_only : bool, default True
            When viewing a VoxelGrid, choose Marching‑Cubes surface
            instead of drawing thousands of cubes.
        """
        # ---------------------------------------------------------
        # 1) Remove previous item, if any
        # ---------------------------------------------------------
        if component in self._items:
            self.gl_view.removeItem(self._items[component])
            del self._items[component]

        # ---------------------------------------------------------
        # 2) Build new vertices / faces
        # ---------------------------------------------------------
        if voxgrid is not None:
            # Extract surface or cube-mesh and bring it into world space
            if surface_only:
                m = voxgrid.marching_cubes.copy()
            else:
                m = voxgrid.as_boxes().copy()

            # Apply the scale/translation so physical size is independent of pitch
            m.apply_transform(voxgrid.transform)

            verts = m.vertices.astype(float)
            faces = m.faces.astype(int)
        elif mesh is not None:
            verts = mesh.vertices.astype(float)
            faces = mesh.faces.astype(int)
        else:
            return  # nothing to show

        # ---------------------------------------------------------
        # 3) Create or update the GLMeshItem for this component
        # ---------------------------------------------------------
        color = self._COLOR.get(component, (0.7, 0.7, 0.7, 1.0))
        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            color=color,
            shader="shaded",
            glOptions="opaque",
        )
        self.gl_view.addItem(item)
        self._items[component] = item

        # ---------------------------------------------------------
        # 4) Bounding boxes
        #     • bei neuem Mesh  -> immer setzen
        #     • bei Re‑Voxelisation -> nur ERWEITERN, nie schrumpfen
        # ---------------------------------------------------------
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)

        if component not in self._bounds_dict or mesh is not None:
            # Erstes Auftreten des Slots ODER neues Mesh ersetzt alte Box
            self._bounds_dict[component] = (vmin, vmax)
        else:
            # Re‑Voxelisation (mesh==None): Box nur vergrößern
            old_min, old_max = self._bounds_dict[component]
            self._bounds_dict[component] = (
                np.minimum(old_min, vmin),
                np.maximum(old_max, vmax),
            )

        # Global bounds across all components
        global_min = np.min(np.vstack([b[0] for b in self._bounds_dict.values()]), axis=0)
        global_max = np.max(np.vstack([b[1] for b in self._bounds_dict.values()]), axis=0)
        global_ext = global_max - global_min

        # --- coordinate axis & camera – update only if scene grows -------
        if self._global_ext is None:
            # first time: take extents as-is
            need_resize = True
        else:
            # Compare new global_ext with previous; allow small epsilon
            need_resize = np.any(global_ext > self._global_ext + 1e-6)

        if need_resize:
            self._global_ext = global_ext.copy()

            # Axis
            if self._axis is None:
                self._axis = GLAxisItem()
                self.gl_view.addItem(self._axis)
            self._axis.setSize(*map(float, global_ext))

            # Camera only when scene expanded (avoid zooming out on re‑voxel)
            global_center = (global_min + global_max) / 2.0
            scene_dist = np.linalg.norm(global_ext) * 1.5
            self.gl_view.setCameraPosition(distance=scene_dist)
            self.gl_view.opts['center'] = Vector(*global_center)

    def update_component(self, component: Component, *,
                         voxgrid=None, mesh=None,
                         surface_only: bool = True):
        """
        Public API: MainWindow calls this whenever one of the three
        CAD components changes (new mesh loaded or workpiece voxel‑
        grid updated).
        """
        self._build_item(component,
                         voxgrid=voxgrid,
                         mesh=mesh,
                         surface_only=surface_only)