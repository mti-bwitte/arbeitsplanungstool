"""
Main application window for the Arbeitsplanungstool.

Responsibilities
----------------
* Presents a single `QToolBar` with
  – *Load File* button (opens file dialog via :class:`core.cadservice.CADService`)
  – voxel pitch (voxel edge length) spin box
  – *Update* button (re_voxelise with current pitch)
* Receives ready signals from :class:`CADService` and embeds a
  :class:`ui.voxel_widget.VoxelWidget` as central widget.

All UI elements are wired via Qt's signal/slot mechanism – no polling.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QToolBar, QPushButton, QDoubleSpinBox
)

from core.cadservice import CADService
from ui.voxel_widget import VoxelWidget

class MainWindow(QMainWindow):
    """
    Top_level window that orchestrates file loading, voxelisation and 3D viewing.
    """

    # --------------------------------------------------------------------- #
    def __init__(self) -> None:
        super().__init__()

        # keep current viewer to access zoom / reset methods from buttons
        self._viewer : VoxelWidget | None = None 

        self.setWindowTitle("Arbeitsplanungstool")
        self.service = CADService(self)

        ## Toolbar with load / pitch / update
        toolbar = QToolBar(self.tr("Main Tools")); self.addToolBar(toolbar)

        # Loading button to load a file
        load_button = QPushButton("Load File")
        toolbar.addWidget(load_button)
        load_button.clicked.connect(self.service.select_file)

        # Pitch input to change voxel edge length
        self.pitch_spin = QDoubleSpinBox(value=1.0, minimum=0.1, singleStep=0.1)
        toolbar.addWidget(self.pitch_spin)
        self.pitch_spin.setSuffix(" mm")
        self.pitch_spin.setEnabled(False) # gets enabled after first load

        # Update button to update pitch
        self.update_button = QPushButton("Update")
        toolbar.addWidget(self.update_button)
        self.update_button.clicked.connect(lambda: self.service.voxelise(self.service.mesh_model, pitch=self.pitch_spin.value()))
        self.update_button.setEnabled(False)

        # Service signals -> slots
        self.service.meshReady.connect(
            lambda m: self.service.voxelise(m, pitch=1.0))
        self.service.voxelReady.connect(self._show_voxels)

    # -- Slots ----------------------------------------------------- #
    def _show_voxels(self, vox):
        widget = VoxelWidget(vox)
        self._viewer = widget
        self.pitch.setEnabled(True)
        self.updatebutton.setEnabled(True)
        self.setCentralWidget(widget)


    def _show_voxels(self, voxgrid) -> None:
        """
        Replace the central widget with a new :class:`VoxelWidget`
        once voxelisation has finished.

        Parameters
        ----------
        voxgrid : trimesh.voxel.VoxelGrid
            The fully voxelised model emitted by :pydata:`voxelReady`.
        """
        viewer = VoxelWidget(voxgrid)
        self._viewer = viewer

        # GUI state updates
        self.pitch_spin.setEnabled(True)
        self.update_button.setEnabled(True)

        self.setCentralWidget(viewer)