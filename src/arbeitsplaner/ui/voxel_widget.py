from typing import Tuple
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
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

    Viewer hotkeys
    --------------
    L       : cycle mesh shader (shaded / balloon / viewNormalColor / normalColor)
    B       : cycle background color (light grey / dark / white)
    Ctrl+P  : save current camera (distance/azimuth/elevation/center/fov)
    Ctrl+O  : restore saved camera exactly
    """

    # Colour per component (R, G, B, A)
    _COLOR = {
        Component.WORKPIECE: (0.8, 0.1, 0.1, 1.0),  # red
        Component.FIXTURE:   (0.2, 0.5, 0.8, 1.0),  # blue-ish
        Component.TOOL:      (0.1, 0.8, 0.1, 1.0),  # green
    }

    # Emits the filled-voxel count each time the mesh is (re)built
    voxelCountChanged = pyqtSignal(int)
    # Emits new world position of the tool centre after every move
    toolMoved = pyqtSignal(np.ndarray)

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

        # --- Viewer appearance controls (lighting/shader & background) ---
        # Available mesh shaders provided by pyqtgraph.opengl
        self._shader_names = [
            "shaded",
            "balloon",
            "viewNormalColor",
            "normalColor",
            "viewRed",
            "normalRed",
            "allRed",   # NEU
        ]
        self._shader_idx: int = 0

        # Background presets to improve contrast in screenshots
        self._bg_colors: list[str] = ["#f0f0f0", "#202020", "w"]
        self._bg_idx: int = 0

        # Layout & OpenGL-View
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.gl_view = gl.GLViewWidget()
        layout.addWidget(self.gl_view)
        # start with a light grey background for better contrast
        self.gl_view.setBackgroundColor(self._bg_colors[self._bg_idx])
        # Forward key events from GLViewWidget to this widget
        self.gl_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.gl_view.installEventFilter(self)
        # Also watch key events on ourselves to be extra safe
        self.installEventFilter(self)

        # Make sure our GL view gets keyboard focus for hotkeys
        self.gl_view.setFocus()

        # Debug toggle for key actions
        self._debug_keys = True

        # Tunables for the "normalRed" mixed mode
        self._mix_normal_red_alpha = 0.75  # more towards red (brighter, less bunt)
        self._mix_normal_red_gamma = 0.85  # brighter midtones
        self._mix_normal_red_ambient = 0.28  # overall brighter base

        # Saturation/strength for viewRed (1.0 = original strong red, lower = less intense/brighter)
        self._viewred_saturation = 0.85


        # Holds the GLMeshItem(s) for every component.
        # WORKPIECE → tuple(red_mesh, green_mesh)
        # others    → single GLMeshItem
        self._items: dict[Component, object] = {}
        self._axis: GLAxisItem | None = None

        # Keep per‑component bounding boxes: {comp: (min_vec, max_vec)}
        self._bounds_dict: dict[Component, tuple[np.ndarray, np.ndarray]] = {}

        # Remember the current global scene size to avoid unnecessary re‑scaling
        self._global_ext: np.ndarray | None = None

        # Viewer starts empty; items are added via update_component(...)
        self.default_distance = 150.0
        self.gl_view.setCameraPosition(distance=self.default_distance)

        # Allow the widget to receive key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Route focus from the container to the GL view so key events land there
        self.setFocusProxy(self.gl_view)

        # Track current tool position in world space
        self.tool_pos = np.zeros(3)
        # Absolute centre of the TOOL mesh in world coordinates (mm)
        self._tool_origin: np.ndarray | None = None

        # Saved camera state (set via Ctrl+P)
        self._saved_cam: dict | None = None

        # Parameters for brightness overlay applied to all modes
        self._cm_gamma: float = 0.75   # <1.0 lifts midtones (brighter)
        self._cm_ambient: float = 0.30 # base fill light

        # Simple multi-light model for brightness overlay
        self._light_dirs = {
            'right': np.array([1.0, 0.0, 0.0]),  # from +X
            'top':   np.array([0.0, 1.0, 0.0]),  # from +Y
            'front': np.array([0.0, 0.0, 1.0]),  # from +Z (not toggled yet)
        }
        self._light_active = {
            'right': True,
            'top':   True,
            'front': True,
        }
        # Emissive factor (0..1): 0 = none, 1 = fully self-lit
        self._emissive = 1.0


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
            if isinstance(self._items[component], tuple):
                for it in self._items[component]:
                    self.gl_view.removeItem(it)
            else:
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
            # Remember the TOOL mesh world‑space centre once
            if component is Component.TOOL and self._tool_origin is None:
                self._tool_origin = verts.mean(axis=0)
        else:
            return  # nothing to show

        # ---------------------------------------------------------
        # 3) Create or update the GLMeshItem for this component
        # ---------------------------------------------------------
        color = self._COLOR.get(component, (0.7, 0.7, 0.7, 1.0))
        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        # Ensure a valid built-in shader at creation time to avoid KeyError on custom modes
        current_mode = self._shader_names[self._shader_idx]
        base_shader = current_mode if current_mode in ("shaded", "balloon", "viewNormalColor", "normalColor") else "shaded"
        item = gl.GLMeshItem(
            meshdata=mesh_data,
            # smoother shading helps with perceived lighting on curved/surface meshes
            smooth=True if (mesh is not None or surface_only) else False,
            color=color,
            shader=base_shader,
            glOptions="opaque",
        )
        # Keep references for color-mode switching
        item._base_color = color
        item._base_meshdata = mesh_data
        self.gl_view.addItem(item)

        # ---------------------------------------------------------
        # 3b) Optional green layer for WORKPIECE ('to_mill')
        # ---------------------------------------------------------
        extra_items: Tuple[gl.GLMeshItem, ...] = ()
        if (
            component is Component.WORKPIECE
            and voxgrid is not None
            and "to_mill" in voxgrid.metadata
        ):
            # Visualise the "to_mill" voxels as a green surface mesh for better visibility
            to_mill = voxgrid.metadata["to_mill"]
            if to_mill.any():
                vg_mill = trimesh.voxel.VoxelGrid(
                    to_mill, transform=voxgrid.transform
                )
                # Render every voxel as a cube so the entire bounding box
                # becomes visible, not just the outer surface.
                green_boxes = vg_mill.as_boxes().copy()
                #green_boxes.apply_transform(vg_mill.transform)

                g_verts = green_boxes.vertices.astype(float)
                g_faces = green_boxes.faces.astype(int)
                g_data = gl.MeshData(vertexes=g_verts, faces=g_faces)
                # Use balloon for voxels in viewRed; otherwise a safe built-in shader
                current_mode = self._shader_names[self._shader_idx]
                if current_mode == "viewRed":
                    overlay_shader = "balloon"
                else:
                    overlay_shader = current_mode if current_mode in ("shaded", "balloon", "viewNormalColor", "normalColor") else "shaded"
                green_item = gl.GLMeshItem(
                    meshdata=g_data,
                    smooth=False,
                    color=(0.0, 1.0, 0.0, 0.15),   # brighter, lighter overlay
                    shader=overlay_shader,
                    glOptions="translucent",         # vivid, glowing voxels
                )
                green_item._base_color = (0.0, 1.0, 0.0, 0.15)
                green_item._base_meshdata = g_data
                self.gl_view.addItem(green_item)
                extra_items = (green_item,)

        self._items[component] = (item, *extra_items) if extra_items else item
        # Reapply colormap/overlay so new/updated items reflect current mode
        mode = self._shader_names[self._shader_idx]
        if mode == "viewRed":
            self._apply_red_normal_coloring_all()
        elif mode == "normalRed":
            self._apply_normal_red_coloring_all()
        elif mode == "shaded":
            self._apply_brightness_overlay_all(mode)
        else:
            # keep built-in appearances (balloon / normalColor / viewNormalColor)
            self.gl_view.update()

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
        
        if mode == "viewRed":
            self._apply_red_normal_coloring_all()
        elif mode == "normalRed":
            self._apply_normal_red_coloring_all()
        elif mode == "allRed":            # NEU
            self._apply_allred_coloring_all()
        elif mode == "shaded":
            self._apply_brightness_overlay_all(mode)
        else:
            self.gl_view.update()

    # -------------------------------------------------------------
    # Appearance helpers: apply shader and toggle background
    # -------------------------------------------------------------
    
    def _apply_allred_coloring_all(self):
        """Uniform, kräftiges Rot nur fürs WORKPIECE-Basismesh."""
        for comp, obj in list(self._items.items()):
            if comp is Component.WORKPIECE:
                target = obj[0] if isinstance(obj, tuple) else obj
                if target is None:
                    continue
                md = getattr(target, "_base_meshdata", None) or target.meshData()
                target._base_meshdata = md
                verts = md.vertexes(); faces = md.faces()
                # Helles Rot: R=1.0, G=B=0.18 (nicht pink, nicht dunkel)
                alpha = (getattr(target, "_base_color", (1,0.06,0.06,1))[3]
                        if len(getattr(target, "_base_color", (1,0.06,0.06,1)))==4 else 1.0)
                r = np.ones((verts.shape[0], 1)); g = np.full((verts.shape[0], 1), 0.18)
                b = np.full((verts.shape[0], 1), 0.18)
                a = np.full((verts.shape[0], 1), alpha)
                vcols = np.hstack([r, g, b, a]).astype(float)
                colored = gl.MeshData(vertexes=verts, faces=faces, vertexColors=vcols)
                try: target.setMeshData(meshdata=colored)
                except TypeError: target.setMeshData(colored)
        self.gl_view.update()
    
    def _apply_shader_all(self):
        """Apply the currently selected shader and reapply overlays/colormaps."""
        mode = self._shader_names[self._shader_idx]
        # Custom red-normal mode: set a base shader, then recolor via vertex colors
        if mode == "viewRed":
            # Base meshes use 'shaded' for good lighting; voxel overlay (workpiece) uses 'balloon'
            for comp, obj in list(self._items.items()):
                if comp is Component.WORKPIECE and isinstance(obj, tuple) and len(obj) >= 1:
                    base_item = obj[0]
                    if hasattr(base_item, "setShader"):
                        base_item.setShader("shaded")
                    # If overlay exists, set it to 'balloon' for the desired look
                    if len(obj) > 1:
                        overlay_item = obj[1]
                        if hasattr(overlay_item, "setShader"):
                            overlay_item.setShader("balloon")
                else:
                    # Other components: use shaded as safe base
                    target_items = obj if isinstance(obj, tuple) else (obj,)
                    for it in target_items:
                        if hasattr(it, "setShader"):
                            it.setShader("shaded")
            self._apply_red_normal_coloring_all()
            return
        elif mode == "allRed":
            base = "shaded"
            for comp, obj in list(self._items.items()):
                if isinstance(obj, tuple):
                    for it in obj:
                        if hasattr(it, "setShader"): it.setShader(base)
                else:
                    if hasattr(obj, "setShader"): obj.setShader(base)
            self._apply_allred_coloring_all()
            return
        elif mode == "normalRed":
            base = "shaded"
            for comp, obj in list(self._items.items()):
                if isinstance(obj, tuple):
                    for it in obj:
                        if hasattr(it, "setShader"):
                            it.setShader(base)
                else:
                    if hasattr(obj, "setShader"):
                        obj.setShader(base)
            self._apply_normal_red_coloring_all()
            return

        # Built-in shaders: apply as-is
        for comp, obj in list(self._items.items()):
            if isinstance(obj, tuple):
                for it in obj:
                    if hasattr(it, "setShader"):
                        it.setShader(mode)
            else:
                if hasattr(obj, "setShader"):
                    obj.setShader(mode)
        # Only apply overlays for shaded; others keep distinct colors
        if mode == "shaded":
            self._apply_brightness_overlay_all(mode)
        else:
            # Do not overlay for balloon / normalColor / viewNormalColor to keep them distinct
            self.gl_view.update()

    def _cycle_shader(self):
        """Cycle through available shaders to change the lighting model."""
        self._shader_idx = (self._shader_idx + 1) % len(self._shader_names)
        mode = self._shader_names[self._shader_idx]
        # Debug print so we can see exactly which mode is active
        if getattr(self, "_debug_keys", False):
            try:
                print(f"[Viewer] Shader mode -> {mode} ({self._shader_idx+1}/{len(self._shader_names)})")
            except Exception:
                pass
        self._apply_shader_all()

    def _cycle_background(self):
        """Toggle the background color to improve perceived exposure."""
        self._bg_idx = (self._bg_idx + 1) % len(self._bg_colors)
        self.gl_view.setBackgroundColor(self._bg_colors[self._bg_idx])
        # Nudge the view to force a repaint on some GL drivers
        self.gl_view.opts['elevation'] = self.gl_view.opts.get('elevation', 0)
        self.gl_view.update()

    def _snapshot_camera(self) -> dict:
        """Return a dict snapshot of the current GLViewWidget camera state."""
        opts = self.gl_view.opts
        center = opts.get('center', Vector(0, 0, 0))
        return {
            'distance': float(opts.get('distance', 100.0)),
            'azimuth': float(opts.get('azimuth', 0.0)),
            'elevation': float(opts.get('elevation', 0.0)),
            'fov': float(opts.get('fov', 60.0)),
            'center': (float(center.x()), float(center.y()), float(center.z())),
        }

    def _apply_camera(self, cam: dict):
        """Apply a previously saved camera snapshot to the GL view."""
        if not cam:
            return
        # Order matters: set position first, then center
        self.gl_view.setCameraPosition(
            distance=cam.get('distance', self.gl_view.opts.get('distance', 100.0)),
            azimuth=cam.get('azimuth', self.gl_view.opts.get('azimuth', 0.0)),
            elevation=cam.get('elevation', self.gl_view.opts.get('elevation', 0.0)),
        )
        self.gl_view.opts['fov'] = cam.get('fov', self.gl_view.opts.get('fov', 60.0))
        cx, cy, cz = cam.get('center', (0.0, 0.0, 0.0))
        self.gl_view.opts['center'] = Vector(cx, cy, cz)
        self.gl_view.update()

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

    # -------------------------------------------------------------
    # Simple keyboard control for the TOOL component
    # WASD keys   → X/Z plane movement
    # Q / E       → vertical movement (Y)
    # -------------------------------------------------------------
    def keyPressEvent(self, event):
        step = 1.0  # mm per key press; adjust as needed
        if getattr(self, "_debug_keys", False):
            try:
                print(f"KEY evt -> key={event.key()} text={repr(event.text())} mods={int(event.modifiers())}")
            except Exception:
                pass

        # Normalize character for robust letter detection (e.g., different layouts)
        ch = (event.text() or "").lower()

        mods = event.modifiers()
        ctrl_down = bool(mods & Qt.KeyboardModifier.ControlModifier)

        # Camera hotkeys: Ctrl+P = save, Ctrl+O = restore
        if ctrl_down and (event.key() == Qt.Key.Key_P or ch == 'p'):
            self._saved_cam = self._snapshot_camera()
            event.accept()
            return
        if ctrl_down and (event.key() == Qt.Key.Key_O or ch == 'o'):
            if self._saved_cam is not None:
                self._apply_camera(self._saved_cam)
            event.accept()
            return


        # Appearance hotkeys: L = cycle shader, B = toggle background
        if event.key() == Qt.Key.Key_L or ch == 'l':
            self._cycle_shader()
            event.accept()
            return
        if event.key() == Qt.Key.Key_B or ch == 'b':
            self._cycle_background()
            event.accept()
            return

        key_map = {
            Qt.Key.Key_A: (-step, 0.0, 0.0),   # left  (X-)
            Qt.Key.Key_D: ( step, 0.0, 0.0),   # right (X+)
            Qt.Key.Key_W: ( 0.0, 0.0,  step),  # forward (Z+)
            Qt.Key.Key_S: ( 0.0, 0.0, -step),  # back    (Z-)
            Qt.Key.Key_E: ( 0.0,  step, 0.0),  # up      (Y+)
            Qt.Key.Key_Q: ( 0.0, -step, 0.0),  # down    (Y-)
        }

        move = key_map.get(event.key())
        if move and Component.TOOL in self._items:
            dx, dy, dz = move
            gl_item = self._items[Component.TOOL]
            if isinstance(gl_item, tuple):  # should not happen, but guard
                gl_item = gl_item[0]
            gl_item.translate(dx, dy, dz)
            self.tool_pos += np.array([dx, dy, dz])
            # Notify listeners (e.g., CADService) about new tool position
            if self._tool_origin is not None:
                abs_pos = self._tool_origin + self.tool_pos
                self.toolMoved.emit(abs_pos)
            else:
                self.toolMoved.emit(self.tool_pos.copy())
            self.gl_view.update()
            event.accept()
            return
        # Not handled here -> ignore so parent/Qt can process
        event.ignore()
        return

    # -------------------------------------------------------------
    # Forward key events from the inner GLViewWidget to keyPressEvent,
    # and handle ShortcutOverride to catch hotkeys before Qt grabs them.
    # Prevent double-triggering of hotkeys by only acting on KeyPress.
    # -------------------------------------------------------------
    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        # Ensure the GL view keeps focus when the mouse enters it
        if obj is self.gl_view and event.type() == QEvent.Type.Enter:
            self.gl_view.setFocus()
            return False

        # Intercept ShortcutOverride to reserve our hotkeys, but do NOT execute actions here
        if obj is self.gl_view:
            if event.type() == QEvent.Type.ShortcutOverride:
                # Reserve keys we handle so no other shortcut steals them
                key = getattr(event, 'key', lambda: None)()
                if key in (
                    Qt.Key.Key_L, Qt.Key.Key_B,
                    Qt.Key.Key_P, Qt.Key.Key_O,
                    Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D,
                    Qt.Key.Key_Q, Qt.Key.Key_E,
                ):
                    event.accept()
                    return True  # consume only the override; real action on KeyPress
                return False

            if event.type() == QEvent.Type.KeyPress:
                # Execute our actions exactly once per physical key press
                self.keyPressEvent(event)
                if event.isAccepted():
                    return True
                return False

        return super().eventFilter(obj, event)
    # ------------------ Color/normal helpers ------------------
    @staticmethod
    def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute per-vertex unit normals from vertices/faces."""
        v = verts
        f = faces.astype(int)
        # face normals
        p0 = v[f[:, 0]]
        p1 = v[f[:, 1]]
        p2 = v[f[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        # avoid zero-length
        fn_len = np.linalg.norm(fn, axis=1) + 1e-12
        fn = fn / fn_len[:, None]
        # accumulate to vertices
        vn = np.zeros_like(v)
        for i in range(3):
            np.add.at(vn, f[:, i], fn)
        # normalize
        l = np.linalg.norm(vn, axis=1) + 1e-12
        vn = vn / l[:, None]
        return vn

    def _apply_brightness_overlay_to_item(self, item: gl.GLMeshItem, mode: str):
        """Apply a per-vertex brightness overlay based on multiple lights
        and emissive term. Works with all shaders; if a shader ignores
        vertex colors (e.g., normalColor), the effect may be limited."""
        md = getattr(item, "_base_meshdata", None)
        base_color = getattr(item, "_base_color", (0.7, 0.7, 0.7, 1.0))
        if md is None:
            md = item.meshData()
            item._base_meshdata = md
        verts = md.vertexes()
        faces = md.faces()
        vn = self._compute_vertex_normals(verts, faces)
        # Compute brightness from all active lights (directional)
        intensity = np.zeros(verts.shape[0], dtype=float)
        for lname, ldir in self._light_dirs.items():
            if self._light_active.get(lname, False):
                lambert = np.clip(vn @ ldir, 0.0, 1.0)
                intensity += lambert
        # Clamp combined light to [0,1] for a stronger, visible toggle effect
        intensity = np.clip(intensity, 0.0, 1.0)
        # Add emissive/self-illumination
        emissive = getattr(self, "_emissive", 0.0)
        intensity = (1.0 - emissive) * intensity + emissive * 1.0
        # Gamma
        gamma = float(getattr(self, "_cm_gamma", 0.75))
        intensity = np.power(intensity, gamma)
        # Ambient fill
        ambient = float(getattr(self, "_cm_ambient", 0.30))
        intensity = np.clip(ambient + (1.0 - ambient) * intensity, 0.0, 1.0)
        base_rgba = np.array(base_color, dtype=float)
        rgb = (base_rgba[None, :3]) * intensity[:, None]
        rgb = np.clip(rgb, 0.0, 1.0)
        a = np.full((verts.shape[0], 1), base_rgba[3] if len(base_rgba) == 4 else 1.0)
        vcols = np.hstack([rgb, a]).astype(float)
        # Create a meshdata with vertex colors but same topology
        colored = gl.MeshData(vertexes=verts, faces=faces, vertexColors=vcols)
        try:
            item.setMeshData(meshdata=colored)
        except TypeError:
            item.setMeshData(colored)

        # --- Fallback: also set a uniform color scaled by mean intensity ---
        # Some pyqtgraph shaders (or versions) may prioritize uniform color.
        # Setting it here guarantees a visible effect even if vertex colors
        # are ignored by the shader.
        mean_int = float(np.mean(intensity))
        base_rgba = np.array(base_color, dtype=float)
        uni_rgb = np.clip(base_rgba[:3] * mean_int, 0.0, 1.0)
        item.setColor((float(uni_rgb[0]), float(uni_rgb[1]), float(uni_rgb[2]), float(base_rgba[3] if len(base_rgba)==4 else 1.0)))



    def _apply_red_normal_coloring_to_item(self, item: gl.GLMeshItem):
        """Apply a red-only variant of normal-based coloring using vertex colors.
        Enhanced contrast for contour/shadow while keeping strong red appearance.
        """
        md = getattr(item, "_base_meshdata", None)
        base_color = getattr(item, "_base_color", (1.0, 0.06, 0.06, 1.0))
        if md is None:
            md = item.meshData()
            item._base_meshdata = md
        verts = md.vertexes()
        faces = md.faces()
        vn = self._compute_vertex_normals(verts, faces)

        # --- Enhanced shading for red mode (brighter) ---
        # Directional lights (sum), view-facing term, and rim highlight
        intensity_l = np.zeros(verts.shape[0], dtype=float)
        for lname, ldir in self._light_dirs.items():
            if self._light_active.get(lname, False):
                lam = np.clip(vn @ ldir, 0.0, 1.0)
                intensity_l += lam
        intensity_l = np.clip(intensity_l, 0.0, 1.0)

        view_f = 0.5 * (vn[:, 2] + 1.0)
        rim = np.power(1.0 - view_f, 2.0)

        # Brighter local tone mapping: higher ambient, mild gamma
        ambient_local = 0.22
        gamma_local = 1.05
        intensity = 0.85 * intensity_l + 0.15 * view_f + 0.20 * rim
        intensity = np.clip(intensity, 0.0, 1.0)
        intensity = ambient_local + (1.0 - ambient_local) * intensity
        intensity = np.power(intensity, gamma_local)

        # Red mapping: strong red with brighter GB to retain shape
        r = np.ones_like(intensity)
        gb = np.clip(0.20 + 0.65 * intensity, 0.0, 1.0)
        g = gb
        b = gb
        # Build RGB then apply saturation blend towards white to reduce intensity
        rgb = np.column_stack([r, g, b]).astype(float)
        sat = float(getattr(self, "_viewred_saturation", 1.0))
        if sat < 1.0:
            rgb = sat * rgb + (1.0 - sat) * 1.0  # blend towards white
        rgb = np.clip(rgb, 0.0, 1.0)
        a = np.full((rgb.shape[0], 1), base_color[3] if len(base_color) == 4 else 1.0)
        vcols = np.hstack([rgb, a]).astype(float)

        colored = gl.MeshData(vertexes=verts, faces=faces, vertexColors=vcols)
        try:
            item.setMeshData(meshdata=colored)
        except TypeError:
            item.setMeshData(colored)

    def _apply_red_normal_coloring_all(self):
        for comp, obj in list(self._items.items()):
            if comp is Component.WORKPIECE:
                if isinstance(obj, tuple) and len(obj) >= 1:
                    it = obj[0]  # base workpiece mesh
                    if not hasattr(it, "_base_meshdata"):
                        it._base_meshdata = it.meshData()
                    if not hasattr(it, "_base_color"):
                        it._base_color = getattr(it, "opts", {}).get("color", (1,1,1,1)) if hasattr(it, "opts") else (1,1,1,1)
                    self._apply_red_normal_coloring_to_item(it)
                elif not isinstance(obj, tuple):
                    it = obj
                    if not hasattr(it, "_base_meshdata"):
                        it._base_meshdata = it.meshData()
                    if not hasattr(it, "_base_color"):
                        it._base_color = getattr(it, "opts", {}).get("color", (1,1,1,1)) if hasattr(it, "opts") else (1,1,1,1)
                    self._apply_red_normal_coloring_to_item(it)
        self.gl_view.update()

    def _apply_normal_red_coloring_to_item(self, item: gl.GLMeshItem):
        """Blend between normalColor-style RGB and our red-normal coloring.
        Gives red-dominant look but retains contour/specular-like variety.
        """
        md = getattr(item, "_base_meshdata", None)
        base_color = getattr(item, "_base_color", (1.0, 0.06, 0.06, 1.0))
        if md is None:
            md = item.meshData()
            item._base_meshdata = md
        verts = md.vertexes()
        faces = md.faces()
        vn = self._compute_vertex_normals(verts, faces)

        # --- normalColor-like mapping (object-space normals to RGB 0..1)
        nc_rgb = (vn * 0.5) + 0.5
        nc_rgb = np.clip(nc_rgb, 0.0, 1.0)

        # --- red-normal mapping (brighter, closer to viewRed, with rim lift)
        view_f = 0.5 * (vn[:, 2] + 1.0)
        view_f = np.power(view_f, 0.80)
        gb = np.clip(0.18 + 0.75 * view_f, 0.0, 1.0)
        red_rgb = np.column_stack([np.ones_like(gb), gb, gb])

        # --- blend (lean more towards red), then rim-lift contours
        a = float(getattr(self, "_mix_normal_red_alpha", 0.75))
        rgb = a * red_rgb + (1.0 - a) * nc_rgb

        # Rim term brightens silhouette a bit to avoid flat areas
        rim = np.power(1.0 - view_f, 2.0)
        rgb = np.clip(rgb + 0.12 * rim[:, None], 0.0, 1.0)

        # local tone curve for brightness/contrast
        amb = float(getattr(self, "_mix_normal_red_ambient", 0.28))
        gam = float(getattr(self, "_mix_normal_red_gamma", 0.85))
        rgb = amb + (1.0 - amb) * rgb
        rgb = np.power(rgb, gam)
        rgb = np.clip(rgb, 0.0, 1.0)

        alpha = base_color[3] if len(base_color) == 4 else 1.0
        vcols = np.column_stack([rgb, np.full((rgb.shape[0], 1), alpha)])
        colored = gl.MeshData(vertexes=verts, faces=faces, vertexColors=vcols)
        try:
            item.setMeshData(meshdata=colored)
        except TypeError:
            item.setMeshData(colored)

    def _apply_normal_red_coloring_all(self):
        for comp, obj in list(self._items.items()):
            if comp is Component.WORKPIECE:
                if isinstance(obj, tuple) and len(obj) >= 1:
                    it = obj[0]
                    if not hasattr(it, "_base_meshdata"):
                        it._base_meshdata = it.meshData()
                    if not hasattr(it, "_base_color"):
                        it._base_color = getattr(it, "opts", {}).get("color", (1,1,1,1)) if hasattr(it, "opts") else (1,1,1,1)
                    self._apply_normal_red_coloring_to_item(it)
                elif not isinstance(obj, tuple):
                    it = obj
                    if not hasattr(it, "_base_meshdata"):
                        it._base_meshdata = it.meshData()
                    if not hasattr(it, "_base_color"):
                        it._base_color = getattr(it, "opts", {}).get("color", (1,1,1,1)) if hasattr(it, "opts") else (1,1,1,1)
                    self._apply_normal_red_coloring_to_item(it)
        self.gl_view.update()

    def _apply_brightness_overlay_all(self, mode: str):
        # Apply brightness overlay to all items
        for comp, obj in list(self._items.items()):
            if isinstance(obj, tuple):
                for it in obj:
                    if not hasattr(it, "_base_meshdata"):
                        it._base_meshdata = it.meshData()
                    if not hasattr(it, "_base_color"):
                        it._base_color = getattr(it, "opts", {}).get("color", (1,1,1,1)) if hasattr(it, "opts") else (1,1,1,1)
                    self._apply_brightness_overlay_to_item(it, mode)
            else:
                it = obj
                if not hasattr(it, "_base_meshdata"):
                    it._base_meshdata = it.meshData()
                if not hasattr(it, "_base_color"):
                    it._base_color = getattr(it, "opts", {}).get("color", (1,1,1,1)) if hasattr(it, "opts") else (1,1,1,1)
                self._apply_brightness_overlay_to_item(it, mode)
        self.gl_view.update()
