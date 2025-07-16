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
    QMainWindow, QToolBar, QPushButton, QDoubleSpinBox,
    QLabel, QComboBox
)
from PyQt6.QtCore import QTimer
import time

from core.cadservice import CADService, Component
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

        # Component selector (Workpiece / Fixture / Tool)
        self.comp_combo = QComboBox()
        self.comp_combo.addItems(["Workpiece", "Fixture", "Tool"])
        toolbar.addWidget(self.comp_combo)
        self.comp_combo.currentIndexChanged.connect(self._update_voxel_button_state)

        # Load CAD button
        self.load_button = QPushButton("Load CAD")
        toolbar.addWidget(self.load_button)
        self.load_button.clicked.connect(self._on_load_clicked)

        # Pitch input to change voxel edge length
        self.pitch_spin = QDoubleSpinBox(value=1.0, minimum=0.1, singleStep=0.1)
        toolbar.addWidget(self.pitch_spin)
        self.pitch_spin.setSuffix(" mm")
        self.pitch_spin.setEnabled(False) # gets enabled after first load

        # Voxelise button
        self.voxel_button = QPushButton("Update")
        toolbar.addWidget(self.voxel_button)
        self.voxel_button.clicked.connect(self._on_voxel_clicked)
        self.voxel_button.setEnabled(False)  # enabled after first mesh

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

        # Service signals -> slots with component enum
        self.service.meshReady.connect(self._on_mesh_ready)
        self.service.voxelReady.connect(self._on_voxel_ready)

    # ------------------------------------------------------------------
    def _selected_component(self) -> Component:
        idx = self.comp_combo.currentIndex()
        return [Component.WORKPIECE, Component.FIXTURE, Component.TOOL][idx]

    # ------------------------------------------------------------------
    def _update_voxel_button_state(self) -> None:
        """
        Enable the Voxelise‑button only when the Workpiece is selected
        *and* a Workpiece mesh is already loaded.
        """
        comp = self._selected_component()
        wp_mesh_loaded = self.service.components[Component.WORKPIECE].mesh is not None
        self.voxel_button.setEnabled(comp is Component.WORKPIECE and wp_mesh_loaded)

    # -- Button handlers ---------------------------------------------- #
    def _on_load_clicked(self):
        comp = self._selected_component()
        self.service.select_file(comp)

    def _on_voxel_clicked(self):
        comp = self._selected_component()
        self._start_voxelisation_timer()
        self.service.voxelise(comp, pitch=self.pitch_spin.value())

    # -- Service signal handlers -------------------------------------- #
    def _on_mesh_ready(self, comp: Component):
        """Viewer update when a mesh is loaded."""
        mesh = self.service.components[comp].mesh
        if self._viewer is None:
            # first time: create an empty viewer (no heavy preprocessing)
            self._viewer = VoxelWidget(None)
            self.setCentralWidget(self._viewer)

        self._viewer.update_component(comp, mesh=mesh)
        # -------------------------------------------------------------
        # Auto‑voxelise Workpiece right after loading (classic workflow)
        # -------------------------------------------------------------
        if comp is Component.WORKPIECE:
            # Start timing _before_ calling voxeliser
            self._start_voxelisation_timer()
            self.service.voxelise(
                Component.WORKPIECE,
                pitch=self.pitch_spin.value()
            )
        # Enable pitch input once any mesh exists
        self.pitch_spin.setEnabled(True)

        # refresh button availability
        self._update_voxel_button_state()

    def _on_voxel_ready(self, comp: Component):
        vg = self.service.components[comp].voxgrid

        # stop voxelisation timer
        if self._vxl_timer.isActive():
            self._vxl_timer.stop()
        if self._vxl_start is not None:
            vxl_elapsed = time.perf_counter() - self._vxl_start
            self.voxelisation_label.setText(
                f"Voxelisation finished: {vxl_elapsed:.1f} s"
            )

        # start visualisation timer
        self._start_visualisation_timer()

        # Ensure viewer exists
        if self._viewer is None:
            self._viewer = VoxelWidget(vg, surface_only=True)
            self.setCentralWidget(self._viewer)

        self._viewer.update_component(comp, voxgrid=vg, surface_only=True)

        # stop visualisation timer
        if self._vis_timer.isActive():
            self._vis_timer.stop()
        if self._vis_start is not None:
            vis_elapsed = time.perf_counter() - self._vis_start
            self.visualisation_label.setText(
                f"Visualisation finished: {vis_elapsed:.1f} s"
            )

    # timing helpers
    # (Removed _on_update_clicked as per instructions)

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