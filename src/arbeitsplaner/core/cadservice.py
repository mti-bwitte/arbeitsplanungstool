# ---------------------------------------------------------------------------
# [bryan-mti annotations]
# This module loads CAD data, voxelises it, and provides utilities
# for the editor / milling simulation. Quick glossary:
#  - pitch: voxel edge length in millimetres (mm per cell)
#  - origin: world coordinate of the voxel corner at index (0,0,0)
#  - occ: boolean 3D array (True = material present)
#  - removed: boolean 3D array (True = already milled away)
#  - to_mill: boolean 3D array (True = still to be milled; UI logic)
#  - transform: 4x4 matrix containing scale (=pitch) and translation (=origin)
# Notes: np.arange(a, b) is half-open (b is exclusive) → use b+1 for inclusive ranges.
# np.meshgrid(..., indexing="ij") builds i,j,k grids matching (x,y,z).
# ---------------------------------------------------------------------------
from pathlib import Path
import trimesh
import os
import numpy as np

from PyQt6.QtCore import (
    QObject,
    pyqtSignal,
    QThreadPool,
    QRunnable,
)  # Qt core classes, signals, thread pool, runnable

# Try to import QtConcurrent; fall back to a thread‑pool runnable
try:
    from PyQt6.QtConcurrent import run as qrun
    _HAVE_CONCURRENT = True
except ImportError:  # QtConcurrent not shipped in some distro builds
    _HAVE_CONCURRENT = False
    _POOL = QThreadPool.globalInstance()

from enum import Enum, auto
from dataclasses import dataclass

# Path objects for convenient file handling

class Component(Enum):
    WORKPIECE = auto()
    FIXTURE   = auto()
    TOOL      = auto()

# Per-slot container: mesh, voxel grid, display name
@dataclass
class CadComponent:
    mesh: trimesh.Trimesh | None = None
    voxgrid: trimesh.voxel.VoxelGrid | None = None
    display: str = ""

class CADService(QObject):

    """
    Service layer that handles file selection, mesh loading, and
    voxelisation.  All heavy work is executed off the GUI thread,
    so the Qt event loop stays responsive.

    Signals
    -------
    meshReady : pyqtSignal(trimesh.Trimesh)
        Fired after a CAD model has been successfully loaded.
    voxelReady : pyqtSignal(trimesh.voxel.VoxelGrid)
        Fired after the loaded mesh has been voxelised.

    Note
    ----
    The public API remains synchronous from the caller's perspective.
    Internally, :pymeth:`voxelise` hands the blocking work to a
    Qt thread‑pool via *QtConcurrent.run* and returns immediately.
    [bryan-mti] Quick overview: the GUI stays responsive; heavy work runs in a worker thread.
    """

    # --- Qt signals ------------------------------------------------------------
    meshReady  = pyqtSignal(Component)  # component ready (mesh loaded)
    voxelReady = pyqtSignal(Component)  # component ready (voxgrid finished)
    voxeliserUsed = pyqtSignal(str)  # e.g. "Open3D CPU" or later "GPU"
    editorReady = pyqtSignal(trimesh.voxel.VoxelGrid)  # Editor should re-render the voxel grid
    voxelsRemoved = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    # Indices of removed voxels → for UI updates / counters / undo

    def __init__(self, parent: QObject | None = None):
        """
        input:
            parent : QObject | None – optional Qt parent for normal
                     ownership / auto‑deletion.  Pass None for a
                     top‑level service object.

        output:
            none
        """
        super().__init__(parent)
    
        # Typed slots for clean signals / API
        self.components: dict[Component, CadComponent] = {
            c: CadComponent(display=c.name.capitalize()) for c in Component
        }

        # track which slot is currently being voxelised (worker uses it)
        self._current_target: Component | None = None


    def select_file(self, target: Component):
        """
        Open a file dialog and load a CAD file into the requested slot.
        Emits meshReady(target) when done.

        input: Component (WORKPIECE, FIXTURE or TOOL)
        output: 
        """
        from PyQt6.QtWidgets import QFileDialog

        # Simple file picker; returns a path string or '' on cancel
        path, _ = QFileDialog.getOpenFileName(
            None,
            f"Select {target.name.capitalize()} file",
            "",
            "CAD files (*.stl *.step *.stp)",  # Note: filter allows *.stl, *.step, *.stp
        )
        if not path:  # user cancelled
            return

        # Load mesh; for STEP, trimesh may triangulate depending on backend
        mesh = trimesh.load(Path(path))
        self.components[target].mesh = mesh
        self.meshReady.emit(target)


    def _load_mesh(self, path:Path):

        """
        input:
            path : pathlib.Path – absolute path to the CAD file
        output:
            none – emits ``meshReady`` with *trimesh.Trimesh*

        Loads the CAD file using *trimesh*; executed on the GUI thread
        because loading is I/O bound and usually quick.
        """
        self.mesh_model = trimesh.load(path)
        self.meshReady.emit(self.mesh_model)

    def voxelise(self, target: Component, pitch: float = 1.0):
        """
        input:
            mesh  : trimesh.Trimesh – triangle mesh to voxelise
            pitch : float           – voxel edge length (same units as mesh)
        output:
            none – result is delivered asynchronously via
            ``voxelReady`` (trimesh.voxel.VoxelGrid)

        Starts voxelisation in a background worker thread using
        *QtConcurrent.run*.  Returns immediately so the GUI thread does
        not block.
        """

        if target is not Component.WORKPIECE:  # currently only voxelising the workpiece
            return

        mesh = self.components[target].mesh
        if mesh is None:  # no mesh loaded yet
            return
        
        # Remember which slot to report in signals
        self._current_target = target

        if _HAVE_CONCURRENT:
            # QtConcurrent available – simplest one‑liner
            qrun(self._voxelise_sync, mesh, pitch)
        else:
            # ----------------------------------------------------------------
            # Fallback: use global QThreadPool if QtConcurrent is unavailable
            # that simply delegates to the existing _voxelise_sync function.
            # ----------------------------------------------------------------
            class _VoxelJob(QRunnable):
                """
                Lightweight wrapper so we can execute `self._voxelise_sync`
                inside the QThreadPool when QtConcurrent is unavailable.
                """

                def __init__(self, fn, m, p):
                    super().__init__()
                    self.fn = fn      # reference to _voxelise_sync
                    self.mesh = m
                    self.pitch = p
                    self.setAutoDelete(True)

                def run(self) -> None:
                    # Call the sync voxeliser; it will emit `voxelReady`
                    # when finished.  Runs in this worker thread.
                    self.fn(self.mesh, self.pitch)

            _POOL.start(_VoxelJob(self._voxelise_sync, mesh, pitch))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _world_to_index(world_xyz: np.ndarray, origin: np.ndarray, pitch: float) -> np.ndarray:
        """
        Parameters
        ----------
        world_xyz : np.ndarray(3,)  – point in millimetres
        origin    : np.ndarray(3,)  – world position of voxel (0,0,0)
        pitch     : float           – mm per voxel

        Returns
        -------
        np.ndarray(3,) integer – voxel indices (ix, iy, iz)
        """
        return np.floor((world_xyz - origin) / pitch).astype(int)  # Example: world=[7,4,2], origin=[0,0,0], pitch=2 → idx=[3,2,1]

    def _voxelise_sync(self, mesh: trimesh.Trimesh, pitch: float) -> None:
        """
        Perform the heavy, blocking voxelisation work.

        input:
            mesh   : trimesh.Trimesh  – triangle soup to convert
            pitch  : float            – voxel edge length (same unit as mesh)

        output:
            none – the resulting ``trimesh.voxel.VoxelGrid`` is delivered
            asynchronously via the ``voxelReady`` signal.

        Details
        -------
        Workflow:
        1.  Convert *mesh* to an Open3D legacy ``TriangleMesh``.
        2.  CPU voxelisation with
            `VoxelGrid.create_from_triangle_mesh()`
        3.  Sparse → dense conversion, **binary_fill_holes** to close voids
        4.  Wrap dense array in a trimesh VoxelGrid and attach attributes.

        The resulting ``voxgrid.user_data`` now contains:

        * 'occ'     – boolean occupancy array (True = material present)
        * 'removed' – boolean machining flag  (True = already milled away)

        Runs inside a worker thread (either QtConcurrent or QRunnable), so
        the GUI thread remains responsive.
        """
        import open3d as o3d
        import numpy as np

        # Local imports so top-level import stays fast and tests can run without O3D

        # --- Step 1 : Trimesh → Open3D TriangleMesh ---------------------------
        # o3d_mesh : open3d.geometry.TriangleMesh
        #   Dense triangle surface in Open3D format.
        #   Example → vertices[0] == [12.3,  7.8, -4.1]
        # Example: mesh.vertices.shape == (V,3), mesh.faces.shape == (F,3)
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
        )

        # --- Step 2 : Voxelisation -------------------------------------------
        # o3d_voxgrid : open3d.geometry.VoxelGrid  (sparse)
        #   Holds only the filled voxels as a hash‑set of 3‑D indices.
        #   Example list → [(12, 7, 3), (12, 7, 4), (13, 7, 3), …]
        # VoxelGrid.create_from_triangle_mesh rasterises the surface; interior still 'hollow' at this point
        o3d_voxgrid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            o3d_mesh, voxel_size=float(pitch)   
        )
        self.voxeliserUsed.emit("Open3D CPU")

        # --- Step 3 : Sparse → dense boolean volume --------------------------
        # indices : np.ndarray[int32]  shape=(n,3)
        #   Each row is (ix, iy, iz) of a filled voxel, i.e. sparse list.
        # indices.shape = (n,3), n = number of filled voxels (sparse set)
        indices = np.asarray([v.grid_index for v in o3d_voxgrid.get_voxels()])
        if indices.size == 0:
            # Minimal empty grid to avoid edge cases
            dense = np.zeros((1, 1, 1), dtype=bool)
        else:
            # dims = max indices + 1  (array size in x,y,z)
            dims = indices.max(axis=0) + 1          # (nx, ny, nz)
            # Sparse → dense occupancy (surface only)
            dense = np.zeros(dims, dtype=bool)
            dense[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # optional: close internal cavities (surface → solid volume)
        from scipy.ndimage import binary_fill_holes

        dense = binary_fill_holes(dense)            # Result: 3D bool array, True = material (interior and surface)

        # -----------------------------------------------------------------
        # Build attribute arrays
        # -----------------------------------------------------------------
        # Attribute bundle for future fields (e.g., temperature, stress, labels)
        # 1) occ  – occupancy: True  → material present
        #                     False → already empty (either never there or milled away)
        occ = dense.astype(bool) 

        # 2) removed – initially everything is NOT removed
        #    During the milling simulation you flip voxels to True.
        removed = np.zeros_like(occ, dtype=bool)

        # Pack attributes into a dictionary so we can extend it later
        # with 'temperature', 'stress', etc.  All arrays share the same
        # shape, so indexing stays trivial.
        attr = {
            "occ":     occ,
            "removed": removed,
        }

        # ------- 4. wrap everything in a Trimesh VoxelGrid --------------
        # pitch is mm per cell → preserves physical dimensions across resolutions
        # origin from O3D grid: world position of voxel corner (0,0,0)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] *= float(pitch)           # uniform scale
        transform[:3, 3] = np.asarray(o3d_voxgrid.origin, dtype=float)  # translation

        # Trimesh VoxelGrid wraps dense bool array + transform
        voxgrid = trimesh.voxel.VoxelGrid(occ, transform=transform)

        # Extra data accessible via vg.metadata['occ'], ['removed'], ['to_mill'], ...
        voxgrid.metadata = attr

        # Put into the correct slot and emit signal
        target = self._current_target or Component.WORKPIECE
        self.components[target].voxgrid = voxgrid

        self.voxelReady.emit(target)

    # ------------------------------------------------------------------
    def turn_to_editorview(self, pitch: float) -> None:
        """
        Build a 'material-to-be-removed' voxel field.

        • Takes the WORKPIECE voxel grid (must already exist).
        • Computes its local bounding box (same shape as occ array).
        • Creates a boolean array `to_mill` that is ``True`` everywhere
          inside the box *except* where the actual workpiece occupies
          material.
        • Stores the new array in ``voxgrid.metadata['to_mill']`` and
          emits it via `editorReady`, so any UI can visualise green
          transparent voxels to mill away.

        Parameters
        ----------
        pitch : float
            Only required if no voxel grid exists yet and we need to
            rasterise.  In the normal workflow the voxel grid is
            already present; pitch is ignored then.

        Notes
        -----
        `removed` voxels will be toggled by `mill_tool` during simulation.
        """
        wp = Component.WORKPIECE
        voxgrid = self.components[wp].voxgrid

        if voxgrid is None:
            # workpiece not yet voxelised – fallback to voxelise now
            self.voxelise(wp, pitch=pitch)
            voxgrid = self.components[wp].voxgrid
            if voxgrid is None:
                return  # still nothing – abort

        # occupancy array (bool) : True = material present
        occ = voxgrid.metadata.get("occ", None)
        if occ is None:
            # Fallback: matrix stores occupancy
            occ = voxgrid.matrix.astype(bool)

        # 'still to mill' = everywhere there is currently no workpiece material
        to_mill = ~occ

        # attach to metadata for downstream visualisation
        voxgrid.metadata["to_mill"] = to_mill

        # emit signal so MainWindow can update the viewer
        self.editorReady.emit(voxgrid)

    # ------------------------------------------------------------------
    def mill_tool(
        self,
        centre_mm: np.ndarray,
        *,
        radius_mm: float,
        length_mm: float,
        axis: str = "y",
    ) -> None:
        """
        Remove (mill) all voxels whose centres lie inside a cylindrical
        tool footprint along the specified axis.

        Parameters
        ----------
            centre_mm : np.ndarray(3,)   – world‑space coord of cutter *centre*
            radius_mm : float            – cutter radius in mm
            length_mm : float            – cutter flute length in mm (along axis)
            axis      : str              – 'x', 'y', or 'z' (tool axis direction)

        Example values:
          centre_mm = np.array([100.0, 50.0, 20.0])
          radius_mm = 5.0
          length_mm = 30.0
          axis      = "y"
        """
        wp = Component.WORKPIECE
        vg = self.components[wp].voxgrid
        if vg is None or "occ" not in vg.metadata:
            return  # nothing to mill

        # occ = occupancy; to_mill = editor logic layer (True = still to be removed)
        occ = vg.metadata["occ"]
        to_mill = vg.metadata.get("to_mill", None)
        if to_mill is None:
            to_mill = ~occ
            vg.metadata["to_mill"] = to_mill

        # pitch: mm per voxel, origin: world position of voxel corner (0,0,0)
        pitch = vg.transform[0, 0]
        origin = vg.transform[:3, 3]

        # Normalize axis and get index
        axis = axis.lower()
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 1)  # axis_idx 0/1/2 → X/Y/Z; default Y for robustness

        # Cylinder endpoints in world coordinates (centre ± half length)
        half_len = length_mm * 0.5
        axis_vec = np.zeros(3)
        axis_vec[axis_idx] = 1.0
        tip_mm = centre_mm - axis_vec * half_len
        top_mm = centre_mm + axis_vec * half_len

        # world → voxel index: floor((p - origin)/pitch)
        tip_idx = self._world_to_index(tip_mm, origin, pitch)
        top_idx = self._world_to_index(top_mm, origin, pitch)

        # r_cells: radius in cells (ceil); centre_idx: voxel index of the centre
        r_cells = int(np.ceil(radius_mm / pitch))
        ranges = [None, None, None]
        centre_idx = self._world_to_index(centre_mm, origin, pitch)
        for ax in range(3):
            if ax == axis_idx:
                start = min(tip_idx[ax], top_idx[ax])
                end   = max(tip_idx[ax], top_idx[ax])
                ranges[ax] = np.arange(start, end + 1)
            else:
                centre = centre_idx[ax]
                ranges[ax] = np.arange(centre - r_cells, centre + r_cells + 1)

        xs, ys, zs = ranges

        # Clamping prevents out-of-bounds indexing
        xs = xs[(xs >= 0) & (xs < occ.shape[0])]
        ys = ys[(ys >= 0) & (ys < occ.shape[1])]
        zs = zs[(zs >= 0) & (zs < occ.shape[2])]

        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            return  # No overlap with grid → nothing to do

        # X,Y,Z are 3D arrays of equal shape; they contain index coordinates of all candidates
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        # Example: X[i,j,k] == xs[i], Y[i,j,k] == ys[j], Z[i,j,k] == zs[k]

        # Compute radial distance perpendicular to axis, relative to centre_mm
        if axis_idx == 0:  # X axis tool
            dy = (Y + 0.5) * pitch + origin[1] - centre_mm[1]
            dz = (Z + 0.5) * pitch + origin[2] - centre_mm[2]
            radial = np.sqrt(dy**2 + dz**2)
        elif axis_idx == 1:  # Y axis
            dx = (X + 0.5) * pitch + origin[0] - centre_mm[0]
            dz = (Z + 0.5) * pitch + origin[2] - centre_mm[2]
            radial = np.sqrt(dx**2 + dz**2)
        else:  # Z axis
            dx = (X + 0.5) * pitch + origin[0] - centre_mm[0]
            dy = (Y + 0.5) * pitch + origin[1] - centre_mm[1]
            radial = np.sqrt(dx**2 + dy**2)

        # +0.5*pitch: safety margin so edge-touching cells are captured
        safe_radius = radius_mm + (pitch * 0.5)
        mask = radial <= safe_radius

        # mask is a 3D boolean array; True = cell centre lies within/on radius
        # Advanced indexing: X[mask], Y[mask], Z[mask] yield 1D lists of hit coordinates
        occ[X[mask], Y[mask], Z[mask]] = False
        to_mill[X[mask], Y[mask], Z[mask]] = False

        # Enables e.g., counters / animated effects / logging
        self.voxelsRemoved.emit(X[mask], Y[mask], Z[mask])

        # Viewer should re-render (material removal visible)
        self.editorReady.emit(vg)

def testfunc():
    return