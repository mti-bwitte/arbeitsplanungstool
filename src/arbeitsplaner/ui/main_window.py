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
    QMainWindow, QToolBar, QPushButton, QDoubleSpinBox, QLabel
)
from PyQt6.QtCore import QTimer
import time
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
        self.update_button.clicked.connect(self._on_update_clicked)
        self.update_button.setEnabled(False)

        # Voxel counter in the status bar
        self.voxel_label = QLabel("Voxel: 0")
        self.statusBar().addPermanentWidget(self.voxel_label)

        # Progress and timer in the status bar
        self.voxelisation_label = QLabel("Voxelisation: idle")
        self.visualisation_label = QLabel("Visualisation: idle")
        self.statusBar().addPermanentWidget(self.voxelisation_label)
        self.statusBar().addPermanentWidget(self.visualisation_label)

        self._vxl_timer = QTimer(self); self._vxl_timer.setInterval(50)
        self._vxl_timer.timeout.connect(self._update_voxelisation_time)
        self._vxl_start = None

        self._vis_timer = QTimer(self); self._vis_timer.setInterval(50)
        self._vis_timer.timeout.connect(self._update_visualisation_time)
        self._vis_start = None

        self.whichvoxelizerused_label = QLabel("Voxelizer used: None")
        self.statusBar().addPermanentWidget(self.whichvoxelizerused_label)

        # Service signals -> slots
        self.service.meshReady.connect(
            lambda m: self.service.voxelise(m, pitch=1.0))
        self.service.voxelReady.connect(self._show_voxels)
        self.service.voxeliserUsed.connect(
            lambda displaytext: self.whichvoxelizerused_label.setText(
                f"Voxelizer used: {displaytext}"
            )
        )

    # -- Slots ----------------------------------------------------- #

    def _show_voxels(self, voxgrid) -> None:
        """
        Replace the central widget with a new :class:`VoxelWidget`
        once voxelisation has finished.

        Parameters
        ----------
        voxgrid : trimesh.voxel.VoxelGrid
            The fully voxelised model emitted by :pydata:`voxelReady`.
        """

        # stop voxelisation timer & freeze duration
        if self._vxl_timer.isActive():
            self._vxl_timer.stop()
        if self._vxl_start is not None:
            vxl_elapsed = time.perf_counter() - self._vxl_start
            self.voxelisation_label.setText(
                f"Voxelisation finished: {vxl_elapsed:.1f} s"
            )
        
        # start visualisation timer
        self._start_visualisation_timer()

        viewer = VoxelWidget(voxgrid, surface_only=True)
        viewer.voxelCountChanged.connect(
            lambda n: self.voxel_label.setText(f"Voxel: {n:,}")
        )
        # show initial count right away
        self.voxel_label.setText(f"Voxel: {viewer.voxel_count:,}")
        self._viewer = viewer

        # GUI state updates
        self.pitch_spin.setEnabled(True)
        self.update_button.setEnabled(True)

        self.setCentralWidget(viewer)

        # stop visualisation timer & freeze duration
        if self._vis_timer.isActive():
            self._vis_timer.stop()
        if self._vis_start is not None:
            vis_elapsed = time.perf_counter() - self._vis_start
            self.visualisation_label.setText(
                f"Visualisation finished: {vis_elapsed:.1f} s"
            )

    # timing helpers
    def _on_update_clicked(self):
        '''Triggered by the Update button.'''
        self._start_voxelisation_timer()
        self.service.voxelise(
            self.service.mesh_model,
            pitch=self.pitch_spin.value()
        )

    # voxelisation timer
    def _start_voxelisation_timer(self):
        self._vxl_start = time.perf_counter()
        # reset visualisation label for the new run
        self.visualisation_label.setText("Visualisation: idle")
        self.voxelisation_label.setText("Voxelisation in progress: 0.0 s")
        if not self._vxl_timer.isActive():
            self._vxl_timer.start()
        
    def _update_voxelisation_time(self):
        if self._vxl_start is None: return
        elapsed = time.perf_counter() - self._vxl_start
        self.visualisation_label.setText(
            "Visualisation: idle"
        )
        self.voxelisation_label.setText(
            f"Voxelisation in progress: {elapsed:.1f} s"
        )
        
    # visualisation timer
    def _start_visualisation_timer(self):
        self._vis_start = time.perf_counter()
        self.visualisation_label.setText("Visualisation in progress: 0.0 s")
        if not self._vis_timer.isActive():
            self._vis_timer.start()

    def _update_visualisation_time(self):
        if self._vis_start is None: return
        elapsed = time.perf_counter() - self._vis_start
        self.visualisation_label.setText(
            f"Visualisation in progress: {elapsed:.1f} s"
        )