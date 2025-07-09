from PyQt6.QtWidgets import QWidget, QVBoxLayout
import numpy as np
import pyqtgraph.opengl as gl

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
    camera_distance : float, optional
        Initial camera distance. If *None*, the distance is automatically
        set to twice the bounding-box diagonal of the voxel grid.
    """


    def __init__(self, voxgrid, parent=None, camera_distance = 100.0):
        
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

        cube_mesh = voxgrid.as_boxes()
        vertices  = np.asarray(cube_mesh.vertices, dtype=float)
        faces     = np.asarray(cube_mesh.faces, dtype=int)

        mesh_data  = gl.MeshData(vertexes=vertices, faces=faces)
        mesh_item  = gl.GLMeshItem(meshdata=mesh_data, smooth=False,
                                   shader="shaded", glOptions="opaque")
        self.gl_view.addItem(mesh_item)

        # Kameraposition

        auto_distance      = np.linalg.norm(cube_mesh.extents) * 2
        self.default_distance = camera_distance or auto_distance
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