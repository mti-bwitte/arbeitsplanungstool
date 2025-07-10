from pathlib import Path
import trimesh
import os

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
            # ----------------------------------------------------------------
            class _VoxelJob(QRunnable):
                def __init__(self, m, p, sig):
                    super().__init__()
                    self.mesh = m
                    self.pitch = p
                    self.sig = sig
                    self.setAutoDelete(True)

                def run(self) -> None:
                    vox = self.mesh.voxelized(self.pitch)
                    self.sig.emit(vox)

            _POOL.start(_VoxelJob(mesh, pitch, self.voxelReady))


    def _voxelise_sync(self, mesh: trimesh.Trimesh, pitch: float) -> None:
        """
        input:
            mesh  : trimesh.Trimesh
            pitch : float
        output:
            none – emits ``voxelReady`` (trimesh.voxel.VoxelGrid)

        CPU‑bound, multi‑threaded voxelisation executed in a background thread.
        The resulting VoxelGrid is sent back to the GUI thread via
        the thread‑safe Qt signal.
        """
        voxgrid = mesh.voxelized(
            pitch,
            method="binvox",          # robust for non‑watertight meshes
            max_threads=os.cpu_count()   # use all CPU cores
        )
        self.voxelReady.emit(voxgrid)
