import sys
import os
import re
import numpy as np
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTreeWidget,
    QLabel, QListWidget, QMessageBox, QFileDialog, QLineEdit, QColorDialog,QDialog,
    QTreeWidgetItem, QTabWidget, QSplitter, QCheckBox, QRadioButton, QButtonGroup
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from gui_backend import DataPlotBackend
from PyQt5.QtCore import Qt
from matplotlib.colors import to_hex
from PyQt5.QtGui import QColor

class SelectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.min_btn = QRadioButton("Min")
        self.max_btn = QRadioButton("Max")
        self.avg_btn = QRadioButton("Average")
        self.x_btn = QRadioButton("Specific Timestep")

        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("Enter integer")
        self.x_input.setFixedWidth(100)
        self.x_input.setEnabled(False)

        # Group radio buttons so only one can be selected
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.min_btn)
        self.button_group.addButton(self.max_btn)
        self.button_group.addButton(self.avg_btn)
        self.button_group.addButton(self.x_btn)

        # Layout for X + input
        x_layout = QHBoxLayout()
        x_layout.addWidget(self.x_btn)
        x_layout.addWidget(self.x_input)

        # Main vertical layout
        layout = QVBoxLayout()
        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addWidget(self.avg_btn)
        layout.addLayout(x_layout)
        layout.addStretch()
        self.setLayout(layout)

        # Connections
        self.button_group.buttonClicked.connect(self.update_input_state)

    def update_input_state(self):
        is_x_selected = self.x_btn.isChecked()
        self.x_input.setEnabled(is_x_selected)

    def get_selection(self):
        """Returns the selected option and value (if 'X' is selected)."""
        if self.min_btn.isChecked():
            return "Min", None
        elif self.max_btn.isChecked():
            return "Max", None
        elif self.avg_btn.isChecked():
            return "Average", None
        elif self.x_btn.isChecked():
            text = self.x_input.text()
            try:
                return "Timestep", int(text)
            except ValueError:
                return "Timestep", None
        return None, None


class AddButtonDialog(QDialog):
    def __init__(self, math_function):
        super().__init__()
        self.setWindowTitle("Add New Button")

        self.math_function = math_function  # Function passed in from frontend

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter button name")

        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.select_color)
        self.selected_color = QColor("lightgray")  # Default color

        # Input field for 'x'
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("Enter value of x")

        # OK/Cancel buttons
        self.ok_button = QPushButton("Add")
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Button Name:"))
        layout.addWidget(self.name_input)
        layout.addWidget(QLabel("Button Color:"))
        layout.addWidget(self.color_button)
        layout.addWidget(QLabel("x Value for Button Action:"))
        layout.addWidget(self.x_input)

        button_row = QHBoxLayout()
        button_row.addWidget(self.ok_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.setLayout(layout)

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color
            self.color_button.setStyleSheet(f"background-color: {color.name()}")

    def get_data(self):
        return (
            self.name_input.text(),
            self.selected_color,
            self.x_input.text()
        )
    

def get_item_hierarchy_text(item):
    texts = []
    while item is not None:
        texts.insert(0, item.text(0))  # Insert at the beginning to get root-first order
        item = item.parent()
    return texts


# --- FRONTEND CLASS ---
class DataPlotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature-Rich Data Plot GUI")
        self.setGeometry(100, 100, 1100, 650)

        self.backend = DataPlotBackend()
        self.data_folder = ""
        self.x_key = None
        self.y_key = None
        self.load_list = []
        # Setup all three tabs
        tabs = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tabs.addTab(tab1, "On the Fly Episode Plotting")
        # tabs.addTab(tab2, "Multiprocess Loading")
        # tabs.addTab(tab3, "Premade Plot Functions")
        self.control_panel = QVBoxLayout()
        tab1.setLayout(self.control_panel)


        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        splitter = QSplitter(Qt.Horizontal)

        left_side_layout = QVBoxLayout()
        left_side = QWidget()
        # Select folder
        folder_btn = QPushButton("Select Folder")
        folder_btn.clicked.connect(self.select_folder)
        left_side_layout.addWidget(folder_btn)

        # Episode file list
        left_side_layout.addWidget(QLabel("Episode Files:"))
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_selected_file)
        left_side_layout.addWidget(self.file_list)

        # Data key list
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Dictionary Viewer")
        left_side_layout.addWidget(self.tree)
        left_side_layout.addWidget(tabs)


        # Tab 1 handles plotting
        # Axis assign buttons
        self.x_button = QPushButton("Set as X Axis")
        self.y_button = QPushButton("Set as Y Axis")
        self.control_panel.addWidget(self.x_button)
        self.control_panel.addWidget(self.y_button)

        self.x_label_display = QLabel("X Axis: None")
        self.y_label_display = QLabel("Y Axis: None")
        self.control_panel.addWidget(self.x_label_display)
        self.control_panel.addWidget(self.y_label_display)

        # Toggle switch for enabling complex options
        self.complex_toggle = QCheckBox("Folder Wide Plotting")
        # self.control_panel.addWidget(self.complex_toggle)

        # The selection widget (initially disabled)
        self.selection_widget = SelectionWidget()
        self.selection_widget.setDisabled(True)
        # self.control_panel.addWidget(self.selection_widget)

        # Connect the toggle to enable/disable the selection widget
        self.complex_toggle.toggled.connect(self.selection_widget.setEnabled)

        # Plot & clear
        plot_button = QPushButton("Plot")
        scatter_button = QPushButton("Scatter")
        clear_button = QPushButton("Clear")
        self.control_panel.addWidget(plot_button)
        self.control_panel.addWidget(scatter_button)
        self.control_panel.addWidget(clear_button)

        # Tab 2 allows the user to specify which pieces of data they want to load into the database for easy
        # plotting, specifically for plotting multiple episodes against one another
        self.selection_widget_loading = SelectionWidget()
        self.tab2_layout = QVBoxLayout()
        self.tab2_layout.addWidget(self.selection_widget_loading)
        load_button_add = QPushButton('Add key to dict')
        self.tab2_layout.addWidget(load_button_add)
        load_button_add.clicked.connect(self.add_to_load_list)
        mid_display = QLabel("Current load list:")
        self.tab2_layout.addWidget(mid_display)
        self.load_list_display = QLabel('[]')
        self.load_list_display.setWordWrap(True)
        self.tab2_layout.addWidget(self.load_list_display)
        start_multiprocess_button = QPushButton('Load data from list')
        self.tab2_layout.addWidget(start_multiprocess_button)
        start_multiprocess_button.clicked.connect(self.load_folder_data)
        clear_button_t2 = QPushButton('Clear Load List')
        self.tab2_layout.addWidget(clear_button_t2)
        clear_button_t2.clicked.connect(self.clear_load_list)
        tab2.setLayout(self.tab2_layout)
        # tab 3 handles custom plotting buttons
        self.tab3_layout = QVBoxLayout()
        add_button = QPushButton("Add Plotting Button")
        add_button.clicked.connect(self.show_add_button_dialog)
        self.tab3_layout.addWidget(add_button)
        config_button = QPushButton("Save additional buttons")
        config_button.clicked.connect(self.save_button_config)
        self.tab3_layout.addWidget(config_button)
        config_button = QPushButton("Load additional buttons")
        config_button.clicked.connect(self.load_button_config)
        self.tab3_layout.addWidget(config_button)
        tab3.setLayout(self.tab3_layout)


        # Plot customization inputs
        left_side_layout.addWidget(QLabel("Plot Title:"))
        self.title_input = QLineEdit()
        self.title_input.textChanged.connect(self.update_labels)
        left_side_layout.addWidget(self.title_input)

        left_side_layout.addWidget(QLabel("X Label:"))
        self.xlabel_input = QLineEdit()
        self.xlabel_input.textChanged.connect(self.update_labels)
        left_side_layout.addWidget(self.xlabel_input)

        left_side_layout.addWidget(QLabel("Y Label:"))
        self.ylabel_input = QLineEdit()
        self.ylabel_input.textChanged.connect(self.update_labels)
        left_side_layout.addWidget(self.ylabel_input)
        # Save plot
        left_side_layout.addWidget(QLabel("Filename to Save (no extension):"))
        self.save_filename_input = QLineEdit()
        left_side_layout.addWidget(self.save_filename_input)

        save_button = QPushButton("Save Plot as PNG")
        save_button.clicked.connect(self.save_plot)
        left_side_layout.addWidget(save_button)
        
        # Plot area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(False)
        left_side.setLayout(left_side_layout)
        splitter.addWidget(left_side)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 500])
        main_layout.addWidget(splitter)
        # main_layout.addWidget(self.canvas)
        
        # Connections
        self.x_button.clicked.connect(self.set_x_axis)
        self.y_button.clicked.connect(self.set_y_axis)
        plot_button.clicked.connect(self.plot_singular)
        scatter_button.clicked.connect(lambda: self.plot_singular(scatter=True))
        clear_button.clicked.connect(self.clear_plot)
        # Add a new section to hold per-line legend inputs
        self.line_legend_inputs_layout = QVBoxLayout()
        self.control_panel.addWidget(QLabel("Legend Labels by Line:"))
        self.control_panel.addLayout(self.line_legend_inputs_layout)

        self.plotted_lines = []  # Store (line2D, QLineEdit) pairs
    @staticmethod
    def square_value(x):
        return x * x
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.data_folder = folder
            self.populate_file_list()

    def add_to_load_list(self):
        # first i need to find the selected thing and add it and the type to the list
        item = self.tree.currentItem()
        if item:
            self.load_list.append([self.selection_widget_loading.get_selection(), get_item_hierarchy_text(item)])
            self.load_list_display.setText(str(self.load_list))

    def clear_load_list(self):
        self.load_list = []
        self.load_list_display.setText(str(self.load_list))

    def load_folder_data(self):
        if self.data_folder and self.load_list:
            self.backend.load_folder(self.data_folder,self.load_list)

    def populate_file_list(self):
        self.file_list.clear()
        episode_files = []
        if os.path.isdir(self.data_folder):
            for fname in os.listdir(self.data_folder):
                if fname.endswith(".pkl"):
                    episode_files.append(fname)
        filenums = [re.findall('\d+',f) for f in episode_files]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        episode_files = episode_files[sorted_inds].tolist()
        for fname in episode_files:
            self.file_list.addItem(fname)
    def load_selected_file(self, item):
        filename = os.path.join(self.data_folder, item.text())
        self.tree.clear()
        try:
            keys = self.backend.load_data(filename, 'timestep_data')
            self.populate_key_list(keys)
        except KeyError:
            keys = self.backend.load_data(filename, 'timestep_list')
            self.populate_key_list(keys)
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_button_config(self):
        print('not implemented yet')

    def load_button_config(self):
        print('also not implemented yet')

    def populate_key_list(self, keys):
        self.populate_tree(self.tree.invisibleRootItem(), keys)
        self.populate_tree(self.tree.invisibleRootItem(), "Timesteps")

    def populate_tree(self, parent_item, data):
        if type(data) is dict:
            for key, value in data.items():
                item = QTreeWidgetItem([str(key)])
                parent_item.addChild(item)
                if isinstance(value, dict):
                    self.populate_tree(item, value)
        else:
            item = QTreeWidgetItem([data])
            parent_item.addChild(item)

    def set_x_axis(self):
        
        item = self.tree.currentItem()
        if item:
            self.x_key = get_item_hierarchy_text(item)
            self.x_label_display.setText(f"X Axis: {self.x_key[-1]}")

    def set_y_axis(self):
        item = self.tree.currentItem()
        if item:
            self.y_key = get_item_hierarchy_text(item)
            self.y_label_display.setText(f"Y Axis: {self.y_key[-1]}")

    def show_add_button_dialog(self):
        dialog = AddButtonDialog(math_function=self.square_value)
        if dialog.exec_():
            name, color, x_value = dialog.get_data()

            if not name or not x_value:
                QMessageBox.warning(self, "Missing Info", "Please enter both a name and x value.")
                return

            try:
                x = float(x_value)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "x must be a number.")
                return

            new_button = QPushButton(name)
            new_button.setStyleSheet(f"background-color: {color.name()};")

            # Bind the button's action to call the static method and print the result
            new_button.clicked.connect(lambda _, val=x: print(self.square_value(val)))

            self.tab3_layout.addWidget(new_button)

    def plot_singular(self, scatter=False):
        if not self.x_key or not self.y_key:
            QMessageBox.warning(self, "Missing Axis", "Please assign both X and Y axes.")
            return

        try:
            x_data, y_data = self.backend.prepare_data_singular(self.x_key, self.y_key)
            # Plot the data
            print(x_data,y_data)
            if scatter:
                line = self.ax.scatter(x_data, y_data, label='placeholder')
                color = to_hex(line.get_facecolor()[0])
            else:
                line = self.ax.plot(x_data, y_data, label='placeholder')[0]
                color = line.get_color()
            self.ax.set_title(self.title_input.text())
            self.ax.set_xlabel(self.xlabel_input.text())
            self.ax.set_ylabel(self.ylabel_input.text())
            # Create input for this line's legend
            
            input_label = QLineEdit()
            input_label.setPlaceholderText(f"Label for line color: {color}")
            input_label.textChanged.connect(self.update_legends)

            # Style with color in title for clarity
            color_label = QLabel(f"Legend (Line color: {color})")
            color_label.setStyleSheet(f"color: {color}; font-weight: bold;")

            # Add label + input to layout
            self.line_legend_inputs_layout.addWidget(color_label)
            self.line_legend_inputs_layout.addWidget(input_label)

            # Store pair
            self.plotted_lines.append((line, input_label))
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", str(e))

    def scatter_singular(self):
        pass

    def update_legends(self):
        # Update each line's legend from the corresponding input
        for line, input_box in self.plotted_lines:
            label = input_box.text().strip()
            if label:
                line.set_label(label)
        self.ax.legend()
        self.canvas.draw()

    def update_labels(self):
        self.ax.set_title(self.title_input.text())
        self.ax.set_xlabel(self.xlabel_input.text())
        self.ax.set_ylabel(self.ylabel_input.text())
        self.canvas.draw()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title(self.title_input.text())
        self.ax.set_xlabel(self.xlabel_input.text())
        self.ax.set_ylabel(self.ylabel_input.text())
        self.ax.grid(False)
        self.canvas.draw()
            # Reset plotted lines
        self.plotted_lines.clear()

        # Clear legend input widgets
        for i in reversed(range(self.line_legend_inputs_layout.count())):
            item = self.line_legend_inputs_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        # self.plotted_lines = []

    def save_plot(self):
        filename = self.save_filename_input.text().strip()
        if not filename:
            QMessageBox.warning(self, "Missing Filename", "Please enter a filename.")
            return

        if not self.data_folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return

        full_path = os.path.join(self.data_folder, filename + ".png")
        self.figure.savefig(full_path)
        QMessageBox.information(self, "Saved", f"Plot saved to:\n{full_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataPlotApp()
    window.show()
    sys.exit(app.exec_())
