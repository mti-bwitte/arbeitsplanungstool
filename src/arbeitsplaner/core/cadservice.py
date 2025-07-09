from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal
import trimesh
from PyQt6.QtWidgets import QFileDialog

class CADService(QObject):

    """
    Service layer that handles file selection, mesh loading and voxelisation.
    Emits ready signals so that the GUI can react without blocking.

    Signals
    -------
    meshReady : pyqtSignal(trimesh.Trimesh)
        Fired after a CAD model has been successfully loaded.
    voxelReady : pyqtSignal(trimesh.voxel.VoxelGrid)
        Fired after the loaded mesh has been voxelised.
    """

    meshReady = pyqtSignal(trimesh.Trimesh)
    voxelReady = pyqtSignal(object)
    
    def select_file(self):

        """
        Open a native file dialog and emit ``meshReady`` once a
        compatible CAD file has been loaded.

        The dialog is modal; the method returns immediately if the user
        cancels.
        """

        path, _ = QFileDialog.getOpenFileName(None, "Datei auswählen", "", "CAD-Dateien (*.stl *.step *.stp)")
        if not path:
            return
        self._load_mesh(Path(path))

    def _load_mesh(self, path:Path):

        """
        Load the selected CAD file with *trimesh* and emit ``meshReady``.
        Parameters
        ----------
        path : pathlib.Path
            Absolute path to the CAD file (STL / STEP).
        """
        self.mesh_model = trimesh.load(path)
        self.meshReady.emit(self.mesh_model)

    def voxelise(self, mesh: trimesh.Trimesh, pitch: float = 1.0):
        """
        Voxelise *mesh* and emit ``voxelReady``.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The triangle mesh to voxelise.
        pitch : float, default 1.0
            Voxel cube edge length in the same unit as *mesh*.

        Notes
        -----
        Uses the default trimesh voxeliser (method='subdivide', fill=True).
        """
        if mesh is None:
            return
        voxgrid = mesh.voxelized(pitch)
        self.voxelReady.emit(voxgrid)
