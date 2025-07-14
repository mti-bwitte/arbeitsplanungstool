from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLAxisItem, GLGridItem

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
        Construct the widget, create an internal `GLViewWidget`, build a
        cube mesh from the voxel grid and set the initial camera
        position.

        Heavy lifting is done in C via `pyqtgraph.opengl`, so the
        constructor is still fast for moderately sized grids.
        """

        super().__init__(parent)

        # Layout & OpenGL-View
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.gl_view = gl.GLViewWidget()
        layout.addWidget(self.gl_view)
        self.gl_view.setBackgroundColor("w")

        # Würfel-Mesh aus VoxelGrid

        if surface_only:
            #Render only the outer surface - far fewer triangles
            surface_mesh = voxgrid.marching_cubes
            vertices = surface_mesh.vertices.astype(float)
            faces = surface_mesh.faces.astype(int)
        else:
            cube_mesh = voxgrid.as_boxes()
            vertices  = np.asarray(cube_mesh.vertices, dtype=float)
            faces     = np.asarray(cube_mesh.faces, dtype=int)

        # choose the mesh whose bb we use for auto camera distance
        mesh_for_extent = surface_mesh if surface_only else cube_mesh

        # Emit voxel count
        voxel_count = int(len(voxgrid.points))
        self.voxel_count = voxel_count
        self.voxelCountChanged.emit(voxel_count)

        mesh_data  = gl.MeshData(vertexes=vertices, faces=faces)
        mesh_item  = gl.GLMeshItem(meshdata=mesh_data, smooth=False,
                                   shader="shaded", glOptions="opaque")
        self.gl_view.addItem(mesh_item)

        # -------------------------------------------------------------
        # Coordinate system
        # -------------------------------------------------------------
        axis = GLAxisItem()
        # Scale axis to match the bounding box of the voxel mesh
        axis.setSize(
            x=float(mesh_for_extent.extents[0]),
            y=float(mesh_for_extent.extents[1]),
            z=float(mesh_for_extent.extents[2]),
        )
        self.gl_view.addItem(axis)

        # Optional: ground grid (XZ‑plane) for orientation
        grid = GLGridItem()
        grid.setSize(
            mesh_for_extent.extents[0],
            mesh_for_extent.extents[2],
        )
        grid.setSpacing(1, 1)  # visual spacing; adjust as needed
        grid.translate(
            -mesh_for_extent.extents[0] / 2,
            -mesh_for_extent.extents[2] / 2,
            -mesh_for_extent.extents[1] / 2,
        )
        self.gl_view.addItem(grid)

        # Kameraposition

        auto_distance      = np.linalg.norm(mesh_for_extent.extents) * 2
        self.default_distance = camera_distance if camera_distance is not None else auto_distance
        self.gl_view.setCameraPosition(distance=self.default_distance)

    # -------------------------------------------------------------
    def zoom(self, factor: float = 0.9) -> None:
       
        """
        Programmatically zoom the camera.

        Parameters
        ----------
        factor : float, default 0.9
            Multiplicative scaling applied to the current camera
            distance.  
            * ``factor < 1``  → zoom in (closer)  
            * ``factor > 1``  → zoom out (further)
        """

        dist = self.gl_view.opts.get("distance", 10.0) * factor
        self.gl_view.setCameraPosition(distance=dist)