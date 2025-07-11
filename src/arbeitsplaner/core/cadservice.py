from pathlib import Path
import trimesh
import os
import numpy as np

from PyQt6.QtCore import (
    QObject,
    pyqtSignal,
    QThreadPool,
    QRunnable,
)

# Try to import QtConcurrent; fall back to a thread‑pool runnable
try:
    from PyQt6.QtConcurrent import run as qrun
    _HAVE_CONCURRENT = True
except ImportError:  # QtConcurrent not shipped in some distro builds
    _HAVE_CONCURRENT = False
    _POOL = QThreadPool.globalInstance()

from PyQt6.QtWidgets import QFileDialog

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
    """

    meshReady = pyqtSignal(trimesh.Trimesh)
    voxelReady = pyqtSignal(object)
    voxeliserUsed = pyqtSignal(str)
    
    def select_file(self):

        """
        input:
            none – opens a modal file dialog
        output:
            none – the method returns immediately; if the user selects a
            file, ``meshReady`` is emitted with *trimesh.Trimesh*.

        Presents a native file dialog and loads an STL / STEP file chosen
        by the user.
        """

        path, _ = QFileDialog.getOpenFileName(None, "Datei auswählen", "", "CAD-Dateien (*.stl *.step *.stp)")
        if not path:
            return
        self._load_mesh(Path(path))

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

    def voxelise(self, mesh: trimesh.Trimesh, pitch: float = 1.0):
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
        if mesh is None:
            return

        if _HAVE_CONCURRENT:
            # QtConcurrent available – simplest one‑liner
            qrun(self._voxelise_sync, mesh, pitch)
        else:
            # ----------------------------------------------------------------
            # Fallback: run in the global QThreadPool via a small QRunnable
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


    #def _voxelise_sync(self, mesh: trimesh.Trimesh, pitch: float) -> None:
    #    """
    #    input:
    #        mesh  : trimesh.Trimesh
    #        pitch : float
    #    output:
    #        none – emits ``voxelReady`` (trimesh.voxel.VoxelGrid)

    #    CPU‑bound, multi‑threaded voxelisation executed in a background thread.
    #    The resulting VoxelGrid is sent back to the GUI thread via
    #    the thread‑safe Qt signal.
    #    """

        #voxgrid = mesh.voxelized(
        #    pitch,
        #    method="ray",          
        #    max_threads=os.cpu_count()   # use all CPU cores
        #).fill()
        #self.voxelReady.emit(voxgrid)

    # NOTE: This method is safe to call from *any* worker thread,
    # either via QtConcurrent or the fallback QRunnable above.
    def _voxelise_sync(self, mesh: trimesh.Trimesh, pitch: float) -> None:
        """
        Open3D-based voxelisation on Metal or CPU, then emit VoxelGrid.
        """
        import open3d as o3d
        import numpy as np

        # ------- 1. Trimesh  ->  Open3D legacy mesh ----------------------------
        # Open3D's stable (legacy) API offers create_from_triangle_mesh(),
        # which works on both CPU and Metal.  We convert the trimesh arrays
        # to Open3D utility vectors.
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
        )

        # ------- 2. Voxelisation ----------------------------------------------
        # create_from_triangle_mesh fills each intersected voxel (surface‑
        # carving).  It is CPU‑only but already much faster than trimesh's
        # subdivide.  For watertight meshes we later run a fill() to close
        # cavities.
        o3d_voxgrid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            o3d_mesh, voxel_size=float(pitch)
        )
        self.voxeliserUsed.emit("Open3D CPU")

        # ------- 3. Convert to dense bool array -------------------------------
        # Open3D stores only filled voxels (sparse).  We reconstruct a dense
        # occupancy grid so that trimesh.VoxelGrid can be used unchanged.

        indices = np.asarray([v.grid_index for v in o3d_voxgrid.get_voxels()])
        if indices.size == 0:
            dense = np.zeros((1, 1, 1), dtype=bool)
        else:
            dims = indices.max(axis=0) + 1          # (nx, ny, nz)
            dense = np.zeros(dims, dtype=bool)
            dense[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # optional: close internal cavities
        from scipy.ndimage import binary_fill_holes

        dense = binary_fill_holes(dense)

        # ------- 4. zurück zu Trimesh-VoxelGrid ------------------------------
        # Build a 4×4 transform whose diagonal encodes the voxel pitch
        transform = np.eye(4)
        transform[0, 0] = transform[1, 1] = transform[2, 2] = float(pitch)

        voxgrid = trimesh.voxel.VoxelGrid(dense, transform=transform)

        self.voxelReady.emit(voxgrid)