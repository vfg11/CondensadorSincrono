"""
main_app.py
=============
Aplicacion principal PySide6. Ventana con 5 botones en orden (parametros
-> linealizar -> reducir -> disenar reguladores -> bateria de tests),
cada uno habilitado solo cuando el anterior esta completo. Guardado y
carga de proyecto desde el menu.

Ejecutar con: python3 main_app.py
"""
import sys, os, json
# Intento siempre anadir las carpetas hermanas al sys.path -- protegido
# con try/except para que sea inofensivo bajo cualquier herramienta de
# empaquetado (Nuitka resuelve los modulos en tiempo de COMPILACION via
# PYTHONPATH, ver .github/workflows/build-windows.yml -- esto de aqui
# es solo para la ejecucion normal desde codigo fuente).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PKG_ROOT = os.path.dirname(_HERE)
    for _d in ['01_model', '02_linearization', '03_design', '04_simulation', '05_gui']:
        _p = os.path.join(_PKG_ROOT, _d)
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
except Exception:
    pass

import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QPushButton, QLabel, QDialog, QTabWidget, QFormLayout,
                                QDoubleSpinBox, QDialogButtonBox, QTextEdit, QMessageBox,
                                QFileDialog, QSpinBox, QProgressDialog, QGroupBox,
                                QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from params_registry import ALL_GROUPS, default_params_dict, build_objects
from workers import WorkerThread, do_linearize, do_reduce, do_design
from plc_battery import run_plc_test, init_plant_plc
from plc_export import build_plc_export
from linear_test import compute_lqi_gain, compute_lqi_observer, linear_step_test


def _canvas_with_toolbar(fig, parent_layout):
    """Anade un canvas matplotlib CON su barra de herramientas estandar
    (zoom, pan, guardar, etc) a un layout, y devuelve el canvas."""
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, None)
    parent_layout.addWidget(toolbar)
    parent_layout.addWidget(canvas)
    return canvas


class ParametersDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parametros del estudio")
        self.resize(520, 480)
        self.spinboxes = {}

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        for group_name, fields in ALL_GROUPS.items():
            page = QWidget()
            form = QFormLayout(page)
            for name, label, default, lo, hi, dec in fields:
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi)
                sb.setDecimals(dec)
                # paso practico para las flechas +/- -- independiente de
                # los decimales mostrados (con 8 decimales, ligar el
                # paso a 10**-dec daria pasos de 1e-7, inutilizable)
                span = hi - lo if np.isfinite(hi - lo) and (hi - lo) < 1e6 else abs(default) + 1.0
                sb.setSingleStep(max(span / 100.0, 10 ** (-dec)))
                sb.setValue(current_params.get(name, default))
                form.addRow(label, sb)
                self.spinboxes[name] = sb
            tabs.addTab(page, group_name)
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {name: sb.value() for name, sb in self.spinboxes.items()}


class OrderDialog(QDialog):
    def __init__(self, suggested, max_order, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Orden del modelo reducido")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Orden sugerido (mayor salto en el espectro de Hankel): {suggested}"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Orden a usar:"))
        self.spin = QSpinBox()
        self.spin.setRange(1, max_order)
        self.spin.setValue(suggested)
        row.addWidget(self.spin)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_order(self):
        return self.spin.value()


class RedesignDialog(QDialog):
    """Ciclo interactivo: ajustar qy/qi (LQI) y Kp/Ki (PI) + tamano de
    escalon, probar, ver grafica, repetir hasta validar. Usa el motor
    RAPIDO (linear_test.py, modelo reducido lineal) para permitir
    iteracion fluida -- la verificacion exhaustiva no lineal es el
    paso 5."""

    def __init__(self, Ar, Br, Cr, Vref0, VREF_MIN, VREF_MAX, Ts, initial, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajuste interactivo de reguladores (LQI + PI)")
        self.resize(980, 780)
        self.Ar, self.Br, self.Cr = Ar, Br, Cr
        self.Vref0, self.VREF_MIN, self.VREF_MAX, self.Ts = Vref0, VREF_MIN, VREF_MAX, Ts
        self.final_result = None
        self.canvas = None
        self._toolbar = None
        self._current_K = self._current_L = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Ajusta los parametros, pulsa 'Probar escalon' para ver la respuesta sobre el "
            "modelo reducido (rapido), repite hasta estar conforme, y 'Validar y continuar'.\n"
            "La verificacion completa sobre la planta no lineal es el paso 5 (bateria de tests)."))

        controls_row = QHBoxLayout()

        lqi_group = QGroupBox("LQI (pesos LQR)")
        lqi_form = QFormLayout(lqi_group)
        self.qy_spin = QDoubleSpinBox(); self.qy_spin.setRange(1e-4, 1e8); self.qy_spin.setDecimals(6)
        self.qy_spin.setValue(initial['qy'])
        self.qi_spin = QDoubleSpinBox(); self.qi_spin.setRange(1e-4, 1e8); self.qi_spin.setDecimals(6)
        self.qi_spin.setValue(initial['qi'])
        lqi_form.addRow("qy (peso salida):", self.qy_spin)
        lqi_form.addRow("qi (peso integral):", self.qi_spin)
        controls_row.addWidget(lqi_group)

        pi_group = QGroupBox("PI")
        pi_form = QFormLayout(pi_group)
        self.kp_spin = QDoubleSpinBox(); self.kp_spin.setRange(0.0, 1e6); self.kp_spin.setDecimals(6)
        self.kp_spin.setValue(initial['Kp'])
        self.ki_spin = QDoubleSpinBox(); self.ki_spin.setRange(0.0, 1e6); self.ki_spin.setDecimals(6)
        self.ki_spin.setValue(initial['Ki'])
        pi_form.addRow("Kp:", self.kp_spin)
        pi_form.addRow("Ki:", self.ki_spin)
        controls_row.addWidget(pi_group)

        test_group = QGroupBox("Prueba de escalon")
        test_form = QFormLayout(test_group)
        self.step_spin = QDoubleSpinBox(); self.step_spin.setRange(-1.0, 1.0); self.step_spin.setDecimals(4)
        self.step_spin.setValue(0.15)
        self.ttotal_spin = QDoubleSpinBox(); self.ttotal_spin.setRange(1.0, 300.0); self.ttotal_spin.setDecimals(1)
        self.ttotal_spin.setValue(15.0)
        test_form.addRow("Tamano del escalon [pu]:", self.step_spin)
        test_form.addRow("Duracion de la prueba [s]:", self.ttotal_spin)
        controls_row.addWidget(test_group)
        layout.addLayout(controls_row)

        self.test_btn = QPushButton("Probar escalon")
        self.test_btn.clicked.connect(self.on_test)
        layout.addWidget(self.test_btn)

        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        final_row = QHBoxLayout()
        self.validate_btn = QPushButton("Validar y continuar")
        self.validate_btn.clicked.connect(self.on_validate)
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        final_row.addWidget(self.validate_btn)
        final_row.addWidget(cancel_btn)
        layout.addLayout(final_row)

        self.on_test()

    def on_test(self):
        qy, qi = self.qy_spin.value(), self.qi_spin.value()
        Kp, Ki = self.kp_spin.value(), self.ki_spin.value()
        step = self.step_spin.value()
        ttotal = self.ttotal_spin.value()

        try:
            K, poles = compute_lqi_gain(self.Ar, self.Br, self.Cr, qy, qi)
        except Exception as e:
            QMessageBox.warning(self, "LQI invalido", f"No se pudo calcular K para estos qy/qi: {e}")
            return
        if np.max(poles.real) >= -1e-8:
            QMessageBox.warning(self, "LQI inestable",
                                 "Estos qy/qi dan un sistema en lazo cerrado INESTABLE -- prueba otros valores.")
            return
        L = compute_lqi_observer(self.Ar, self.Cr)

        log_lqi = linear_step_test('LQI', self.Ar, self.Br, self.Cr, self.Ts, ttotal, step, 1.0,
                                    self.Vref0, self.VREF_MIN, self.VREF_MAX, K=K, L=L)
        log_pi = linear_step_test('PI', self.Ar, self.Br, self.Cr, self.Ts, ttotal, step, 1.0,
                                   self.Vref0, self.VREF_MIN, self.VREF_MAX, Kp=Kp, Ki=Ki)
        self._current_K, self._current_L = K, L

        if self.canvas is not None:
            self.canvas.setParent(None)
            self._toolbar.setParent(None)
        fig = Figure(figsize=(9, 4.3))
        axQ = fig.add_subplot(1, 2, 1)
        axV = fig.add_subplot(1, 2, 2)
        axQ.plot(log_lqi['t'], log_lqi['Q'], color='tab:blue', linewidth=1.4, label='LQI')
        axQ.plot(log_pi['t'], log_pi['Q'], color='tab:red', linewidth=1.4, label='PI')
        axQ.axhline(step, color='gray', ls='--', linewidth=0.8)
        axQ.set_title(f"Q (desviacion respecto al equilibrio): 0 -> {step:.4g}")
        axQ.set_xlabel('t [s]'); axQ.set_ylabel('Q [pu]'); axQ.grid(alpha=0.3); axQ.legend()
        axV.plot(log_lqi['t'], log_lqi['Vref'], color='tab:blue', linewidth=1.4, label='LQI')
        axV.plot(log_pi['t'], log_pi['Vref'], color='tab:red', linewidth=1.4, label='PI')
        axV.axhline(self.VREF_MAX, color='black', ls=':', linewidth=1)
        axV.axhline(self.VREF_MIN, color='black', ls=':', linewidth=1)
        axV.set_title("Vref aplicado"); axV.set_xlabel('t [s]'); axV.set_ylabel('Vref [pu]')
        axV.grid(alpha=0.3); axV.legend()
        fig.tight_layout()
        self.canvas = FigureCanvas(fig)
        self._toolbar = NavigationToolbar(self.canvas, None)
        self.canvas_layout.addWidget(self._toolbar)
        self.canvas_layout.addWidget(self.canvas)

        self.info_label.setText(f"Polos del LQI en lazo cerrado (parte real): {np.sort(poles.real).round(3)}")

    def on_validate(self):
        if self._current_K is None:
            QMessageBox.warning(self, "Sin prueba", "Prueba el escalon al menos una vez antes de validar.")
            return
        self.final_result = dict(K=self._current_K, L=self._current_L,
                                  qy=self.qy_spin.value(), qi=self.qi_spin.value(),
                                  Kp=self.kp_spin.value(), Ki=self.ki_spin.value(), Ts=self.Ts)
        self.accept()


class BatteryConfigDialog(QDialog):
    """Configura los 3 ensayos de la bateria antes de lanzarla: puntos
    de escalon (Q0, tamano, duracion), falla (duracion, magnitud,
    instante, duracion total), y perturbacion de red (tramos relativos
    de Einf, duracion de cada tramo, instante de inicio)."""

    def __init__(self, Q0_OP, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar bateria de tests")
        self.resize(650, 650)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # --- Escalon de consigna ---
        step_page = QWidget()
        step_layout = QVBoxLayout(step_page)
        step_layout.addWidget(QLabel("Puntos de operacion a probar (Q0, tamano del escalon, duracion):"))
        self.step_table = QTableWidget(4, 3)
        self.step_table.setHorizontalHeaderLabels(["Q0 [pu]", "Escalon [pu]", "Duracion [s]"])
        self.step_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        default_points = [(-0.15, 0.15, 30.0), (Q0_OP, 0.15, 30.0), (0.60, 0.15, 30.0), (0.70, 0.10, 30.0)]
        for row, (q0, step, dur) in enumerate(default_points):
            self.step_table.setItem(row, 0, QTableWidgetItem(f"{q0:g}"))
            self.step_table.setItem(row, 1, QTableWidgetItem(f"{step:g}"))
            self.step_table.setItem(row, 2, QTableWidgetItem(f"{dur:g}"))
        step_layout.addWidget(self.step_table)
        step_btn_row = QHBoxLayout()
        btn_add_step = QPushButton("Anadir fila")
        btn_add_step.clicked.connect(lambda: self._add_row(self.step_table, ["0.0", "0.1", "20.0"]))
        btn_del_step = QPushButton("Quitar fila seleccionada")
        btn_del_step.clicked.connect(lambda: self._del_row(self.step_table))
        step_btn_row.addWidget(btn_add_step); step_btn_row.addWidget(btn_del_step)
        step_layout.addLayout(step_btn_row)
        tabs.addTab(step_page, "Escalon de consigna")

        # --- Falla ---
        fault_page = QWidget()
        fault_form = QFormLayout(fault_page)
        self.fault_duration = QDoubleSpinBox(); self.fault_duration.setRange(0.001, 10.0)
        self.fault_duration.setDecimals(4); self.fault_duration.setValue(0.15)
        self.fault_gfault = QDoubleSpinBox(); self.fault_gfault.setRange(0.0, 1000.0)
        self.fault_gfault.setDecimals(4); self.fault_gfault.setValue(25.0)
        self.fault_tfault = QDoubleSpinBox(); self.fault_tfault.setRange(0.0, 100.0)
        self.fault_tfault.setDecimals(3); self.fault_tfault.setValue(1.0)
        self.fault_ttotal = QDoubleSpinBox(); self.fault_ttotal.setRange(1.0, 300.0)
        self.fault_ttotal.setDecimals(2); self.fault_ttotal.setValue(20.0)
        fault_form.addRow("Duracion de la falla [s]:", self.fault_duration)
        fault_form.addRow("Magnitud (Gfault, admitancia shunt) [pu]:", self.fault_gfault)
        fault_form.addRow("Instante de inicio [s]:", self.fault_tfault)
        fault_form.addRow("Duracion total del ensayo [s]:", self.fault_ttotal)
        tabs.addTab(fault_page, "Falla")

        # --- Perturbacion de red ---
        volt_page = QWidget()
        volt_layout = QVBoxLayout(volt_page)
        volt_layout.addWidget(QLabel("Tramos encadenados de tension de red (variacion relativa sobre Einf):"))
        self.volt_table = QTableWidget(4, 1)
        self.volt_table.setHorizontalHeaderLabels(["Escalon relativo Einf"])
        self.volt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row, val in enumerate([-0.05, 0.05, -0.10, 0.10]):
            self.volt_table.setItem(row, 0, QTableWidgetItem(f"{val:g}"))
        volt_layout.addWidget(self.volt_table)
        volt_btn_row = QHBoxLayout()
        btn_add_volt = QPushButton("Anadir fila")
        btn_add_volt.clicked.connect(lambda: self._add_row(self.volt_table, ["0.0"]))
        btn_del_volt = QPushButton("Quitar fila seleccionada")
        btn_del_volt.clicked.connect(lambda: self._del_row(self.volt_table))
        volt_btn_row.addWidget(btn_add_volt); volt_btn_row.addWidget(btn_del_volt)
        volt_layout.addLayout(volt_btn_row)
        volt_extra_form = QFormLayout()
        self.volt_seg = QDoubleSpinBox(); self.volt_seg.setRange(0.5, 120.0)
        self.volt_seg.setDecimals(2); self.volt_seg.setValue(10.0)
        self.volt_t0 = QDoubleSpinBox(); self.volt_t0.setRange(0.0, 60.0)
        self.volt_t0.setDecimals(2); self.volt_t0.setValue(1.0)
        volt_extra_form.addRow("Duracion de cada tramo [s]:", self.volt_seg)
        volt_extra_form.addRow("Instante del primer escalon [s]:", self.volt_t0)
        volt_layout.addLayout(volt_extra_form)
        tabs.addTab(volt_page, "Perturbacion de red")

        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Ejecutar bateria")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_row(self, table, defaults):
        row = table.rowCount()
        table.insertRow(row)
        for col, val in enumerate(defaults):
            table.setItem(row, col, QTableWidgetItem(val))

    def _del_row(self, table):
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _read_table(self, table, ncols):
        rows = []
        for r in range(table.rowCount()):
            try:
                vals = [float(table.item(r, c).text()) for c in range(ncols)]
            except (ValueError, AttributeError):
                continue
            rows.append(vals)
        return rows

    def get_config(self):
        step_rows = self._read_table(self.step_table, 3)
        points = [(f"Q0={q0:g}", q0, step, dur) for q0, step, dur in step_rows]
        volt_rows = self._read_table(self.volt_table, 1)
        schedule = [v[0] for v in volt_rows]
        return dict(
            points=points,
            fault=dict(duration=self.fault_duration.value(), Gfault=self.fault_gfault.value(),
                       t_fault=self.fault_tfault.value(), t_total=self.fault_ttotal.value()),
            voltage=dict(schedule=schedule, seg=self.volt_seg.value(), t_event0=self.volt_t0.value()),
        )


class BatteryResultsDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resultados de la bateria de tests")
        self.resize(1150, 900)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_step_tab(results['step']), "Escalon de consigna")
        tabs.addTab(self._build_fault_tab(results['fault']), "Falla 150ms")
        tabs.addTab(self._build_voltage_tab(results['voltage']), "Perturbacion de red")
        layout.addWidget(tabs)
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _build_step_tab(self, step_results):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        fig = Figure(figsize=(11, 9))
        colors = dict(PI='tab:red', LQI='tab:blue', Rele='tab:green')
        n_points = len(step_results)
        for i, (label, entry) in enumerate(step_results.items()):
            per_controller = entry['data']
            Qbase, target = entry['Qbase'], entry['target']
            axQ = fig.add_subplot(n_points, 3, i * 3 + 1)
            axV = fig.add_subplot(n_points, 3, i * 3 + 2)
            axT = fig.add_subplot(n_points, 3, i * 3 + 3)
            for name, r in per_controller.items():
                axQ.plot(r['t'], r['Q'], color=colors[name], linewidth=1.2, label=name)
                axV.plot(r['t'], r['Vref'], color=colors[name], linewidth=1.2, label=name)
                axT.plot(r['t'], r['Vt'], color=colors[name], linewidth=1.2, label=name)
            axQ.axhline(target, color='gray', ls='--', linewidth=0.8)
            axQ.set_title(f"{label}: Q {Qbase:.4g} -> {target:.4g}", fontsize=8)
            axQ.grid(alpha=0.3); axQ.tick_params(labelsize=6)
            axV.set_title("Vref", fontsize=8); axV.grid(alpha=0.3); axV.tick_params(labelsize=6)
            axT.set_title("Vt", fontsize=8); axT.grid(alpha=0.3); axT.tick_params(labelsize=6)
            if i == 0:
                axQ.legend(fontsize=6)
        fig.tight_layout()
        _canvas_with_toolbar(fig, page_layout)
        return page

    def _build_fault_tab(self, fault_results):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        fig = Figure(figsize=(9, 8))
        colors = dict(PI='tab:red', LQI='tab:blue', Rele='tab:green')
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        for name, r in fault_results.items():
            ax1.plot(r['t'], r['Q'], color=colors[name], linewidth=1.3, label=name)
            ax2.plot(r['t'], r['Vref'], color=colors[name], linewidth=1.3, label=name)
            ax3.plot(r['t'], r['Vt'], color=colors[name], linewidth=1.3, label=name)
        ax1.set_ylabel('Q [pu]'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
        ax2.set_ylabel('Vref [pu]'); ax2.grid(alpha=0.3)
        ax3.set_ylabel('Vt [pu]'); ax3.set_xlabel('t [s]'); ax3.grid(alpha=0.3)
        fig.tight_layout()
        _canvas_with_toolbar(fig, page_layout)
        return page

    def _build_voltage_tab(self, voltage_results):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        fig = Figure(figsize=(9, 9))
        colors = dict(PI='tab:red', LQI='tab:blue', Rele='tab:green')
        ax0 = fig.add_subplot(4, 1, 1)
        ax1 = fig.add_subplot(4, 1, 2)
        ax2 = fig.add_subplot(4, 1, 3)
        ax3 = fig.add_subplot(4, 1, 4)
        any_r = next(iter(voltage_results.values()))
        ax0.plot(any_r['t'], any_r['Einf'], color='black', linewidth=1.2)
        ax0.set_ylabel('Einf [pu]'); ax0.grid(alpha=0.3)
        for name, r in voltage_results.items():
            ax1.plot(r['t'], r['Q'], color=colors[name], linewidth=1.2, label=name)
            ax2.plot(r['t'], r['Vref'], color=colors[name], linewidth=1.2, label=name)
            ax3.plot(r['t'], r['Vt'], color=colors[name], linewidth=1.2, label=name)
        ax1.set_ylabel('Q [pu]'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
        ax2.set_ylabel('Vref [pu]'); ax2.grid(alpha=0.3)
        ax3.set_ylabel('Vt [pu]'); ax3.set_xlabel('t [s]'); ax3.grid(alpha=0.3)
        fig.tight_layout()
        _canvas_with_toolbar(fig, page_layout)
        return page


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Estudio de condensador sincrono -- LQI / PI / Rele")
        self.resize(560, 420)

        self.params = default_params_dict()
        self.lin_result = None
        self.red_result = None
        self.design_result = None
        self.battery_result = None
        self._thread = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.status_labels = {}
        self.buttons = {}

        def add_step(key, text):
            row = QHBoxLayout()
            btn = QPushButton(text)
            status = QLabel("pendiente")
            status.setFixedWidth(90)
            row.addWidget(btn)
            row.addWidget(status)
            layout.addLayout(row)
            self.buttons[key] = btn
            self.status_labels[key] = status
            return btn

        add_step('params', "1. Parametros de la maquina...").clicked.connect(self.on_params)
        add_step('lin', "2. Linealizacion simbolica").clicked.connect(self.on_linearize)
        add_step('red', "3. Reduccion balanceada").clicked.connect(self.on_reduce)
        add_step('design', "4. Disenar reguladores (LQI + PI)").clicked.connect(self.on_design)
        add_step('battery', "5. Bateria de tests").clicked.connect(self.on_battery)
        add_step('export_plc', "Exportar implementacion PLC (LQI)...").clicked.connect(self.on_export_plc)

        save_row = QHBoxLayout()
        btn_save = QPushButton("Guardar proyecto...")
        btn_save.clicked.connect(self.on_save)
        btn_load = QPushButton("Cargar proyecto...")
        btn_load.clicked.connect(self.on_load)
        save_row.addWidget(btn_save)
        save_row.addWidget(btn_load)
        layout.addLayout(save_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)

        self._update_button_states()
        self.status_labels['params'].setText("por defecto")

    def _log(self, msg):
        self.log.append(str(msg))

    def _update_button_states(self):
        self.buttons['lin'].setEnabled(True)
        self.buttons['red'].setEnabled(self.lin_result is not None)
        self.buttons['design'].setEnabled(self.red_result is not None)
        self.buttons['battery'].setEnabled(self.design_result is not None)
        self.buttons['export_plc'].setEnabled(self.design_result is not None)

    def _run_worker(self, fn, kwargs, on_success, title):
        dlg = QProgressDialog(title, None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.show()
        self._thread = WorkerThread(fn, kwargs)
        self._thread.progress.connect(self._log)
        self._thread.finished_ok.connect(lambda r: (dlg.close(), on_success(r)))
        self._thread.failed.connect(lambda msg: (dlg.close(), self._on_error(msg)))
        self._thread.start()

    def _on_error(self, msg):
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg[:2000])

    def on_params(self):
        dlg = ParametersDialog(self.params, self)
        if dlg.exec() == QDialog.Accepted:
            self.params = dlg.get_params()
            self.status_labels['params'].setText("ajustados")
            self.lin_result = self.red_result = self.design_result = self.battery_result = None
            for k in ['lin', 'red', 'design', 'battery']:
                self.status_labels[k].setText("pendiente")
            self._update_button_states()
            self._log("Parametros actualizados.")

    def on_linearize(self):
        objs = build_objects(self.params)
        self._run_worker(do_linearize, dict(objs=objs), self._on_linearize_done,
                          "Linealizando (puede tardar ~10s)...")

    def _on_linearize_done(self, result):
        self.lin_result = result
        self.status_labels['lin'].setText("completo")
        active = [k for k, v in result['flags'].items() if v]
        self._log(f"Linealizacion OK. Vref0={result['Vref0']:.4f}. Flags activos: {active}")
        self._update_button_states()

    def on_reduce(self):
        q_idx = list(self.lin_result['output_names']).index('Qdeliv_m')
        C_row = self.lin_result['C'][q_idx:q_idx + 1, :]
        A, B = self.lin_result['A'], self.lin_result['B']
        from workers import compute_hankel_spectrum, suggest_order
        hsv = compute_hankel_spectrum(A, B, C_row)
        suggested = suggest_order(hsv)
        dlg = OrderDialog(suggested, A.shape[0] - 1, self)
        if dlg.exec() != QDialog.Accepted:
            return
        order = dlg.get_order()
        self._run_worker(do_reduce, dict(A=A, B=B, C_row=C_row, order=order),
                          self._on_reduce_done, "Reduciendo...")

    def _on_reduce_done(self, result):
        self.red_result = result
        self.status_labels['red'].setText(f"orden {result['order_used']}")
        self._log(f"Reduccion OK. Orden={result['order_used']} (sugerido {result['suggested_order']}). "
                   f"DC completo={result['dc_full']:.6f} DC reducido={result['dc_red']:.6f}")
        self._update_button_states()

    def on_design(self):
        Ts = self.params['Ts']
        self._run_worker(do_design,
                          dict(Ar=self.red_result['Ar'], Br=self.red_result['Br'],
                               Cr=self.red_result['Cr'], Ts=Ts),
                          self._on_initial_design_done, "Buscando un punto de partida razonable...")

    def _on_initial_design_done(self, result):
        self._log(f"Punto de partida: LQI qy={result['qy']} qi={result['qi']}. "
                   f"PI Kp={result['Kp']} Ki={result['Ki']}. Ahora ajusta y prueba el escalon.")
        dlg = RedesignDialog(self.red_result['Ar'], self.red_result['Br'], self.red_result['Cr'],
                              self.lin_result['Vref0'], self.params['VREF_MIN'], self.params['VREF_MAX'],
                              Ts=result['Ts'], initial=result, parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.final_result is not None:
            self.design_result = dlg.final_result
            self.status_labels['design'].setText("completo (validado)")
            self._log(f"Reguladores validados por el usuario. LQI: qy={dlg.final_result['qy']} "
                       f"qi={dlg.final_result['qi']}. PI: Kp={dlg.final_result['Kp']} "
                       f"Ki={dlg.final_result['Ki']}.")
        else:
            self._log("Ajuste de reguladores cancelado -- paso 4 sigue pendiente.")
        self._update_button_states()

    def on_export_plc(self):
        objs = build_objects(self.params)
        try:
            doc = build_plc_export(self.params, objs, self.lin_result, self.red_result,
                                    self.design_result)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo generar el documento: {e}")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar implementacion PLC", "",
                                               "Texto (*.txt)")
        if not path:
            return
        if not path.endswith('.txt'):
            path += '.txt'
        with open(path, 'w') as f:
            f.write(doc)
        self._log(f"Documento de implementacion PLC exportado a {path}")

    def on_battery(self):
        if self.battery_result is not None:
            box = QMessageBox(self)
            box.setWindowTitle("Bateria ya ejecutada")
            box.setText("Ya hay resultados de una ejecucion anterior de la bateria de tests.")
            recompute_btn = box.addButton("Volver a calcular", QMessageBox.ActionRole)
            reuse_btn = box.addButton("Ver los datos ya obtenidos", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked == reuse_btn:
                dlg = BatteryResultsDialog(self.battery_result, self)
                dlg.exec()
                return
            elif clicked != recompute_btn:
                return  # cancelado

        cfg_dlg = BatteryConfigDialog(self.params['Q0_OP'], self)
        if cfg_dlg.exec() != QDialog.Accepted:
            return
        battery_config = cfg_dlg.get_config()
        if not battery_config['points']:
            QMessageBox.warning(self, "Sin puntos", "Anade al menos un punto de escalon.")
            return

        objs = build_objects(self.params)
        Ts = self.params['Ts']
        Q0_OP = self.params['Q0_OP']
        Ar, Br, Cr = self.red_result['Ar'], self.red_result['Br'], self.red_result['Cr']
        K, L = self.design_result['K'], self.design_result['L']
        Kp, Ki = self.design_result['Kp'], self.design_result['Ki']
        kwargs = dict(objs=objs, Ts=Ts, Q0_OP=Q0_OP, Ar=Ar, Br=Br, Cr=Cr, K=K, L=L, Kp=Kp, Ki=Ki,
                       battery_config=battery_config)
        self._run_worker(_run_full_battery, kwargs, self._on_battery_done,
                          "Ejecutando bateria de tests (puede tardar varios minutos)...")

    def _on_battery_done(self, result):
        self.battery_result = result
        self.status_labels['battery'].setText("completo")
        self._log("Bateria de tests completa.")
        dlg = BatteryResultsDialog(result, self)
        dlg.exec()

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar proyecto", "", "Proyecto (*.npz)")
        if not path:
            return
        if not path.endswith('.npz'):
            path += '.npz'
        payload = dict(params_json=json.dumps(self.params))
        if self.lin_result:
            payload.update({f'lin_{k}': v for k, v in self.lin_result.items() if k != 'flags'})
            payload['lin_flags_json'] = json.dumps({k: bool(v) for k, v in self.lin_result['flags'].items()})
        if self.red_result:
            payload.update({f'red_{k}': v for k, v in self.red_result.items()})
        if self.design_result:
            payload.update({f'des_{k}': v for k, v in self.design_result.items()})
        np.savez(path, **payload)
        self._log(f"Proyecto guardado en {path}")

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar proyecto", "", "Proyecto (*.npz)")
        if not path:
            return
        d = np.load(path, allow_pickle=True)
        self.params = json.loads(str(d['params_json']))
        if 'lin_A' in d:
            flags = json.loads(str(d['lin_flags_json'])) if 'lin_flags_json' in d else {}
            self.lin_result = dict(A=d['lin_A'], B=d['lin_B'], C=d['lin_C'], D=d['lin_D'],
                                    state_names=d['lin_state_names'], output_names=d['lin_output_names'],
                                    Vref0=float(d['lin_Vref0']), flags=flags)
            self.status_labels['lin'].setText("completo (cargado)")
        if 'red_Ar' in d:
            self.red_result = dict(Ar=d['red_Ar'], Br=d['red_Br'], Cr=d['red_Cr'], Dr=d['red_Dr'],
                                    hsv=d['red_hsv'], suggested_order=int(d['red_suggested_order']),
                                    order_used=int(d['red_order_used']),
                                    dc_full=float(d['red_dc_full']), dc_red=float(d['red_dc_red']))
            self.status_labels['red'].setText(f"orden {self.red_result['order_used']} (cargado)")
        if 'des_K' in d:
            self.design_result = dict(K=d['des_K'], L=d['des_L'], qy=float(d['des_qy']),
                                       qi=float(d['des_qi']), Kp=float(d['des_Kp']),
                                       Ki=float(d['des_Ki']), Ts=float(d['des_Ts']))
            self.status_labels['design'].setText("completo (cargado)")
        self.status_labels['params'].setText("cargados")
        self._update_button_states()
        self._log(f"Proyecto cargado desde {path}")


def _run_one(cname, gains, objs, Q0, step, Ts, t_total, t_event=1.0):
    _, _, _, _, Qbase = init_plant_plc(objs, Q0)
    if cname == 'LQI':
        gains['Qeq'] = Qbase
    qref = lambda t: Qbase + (step if t >= t_event else 0.0)
    log, Qbase2, Vref0 = run_plc_test(cname, gains, objs, Q0, qref, Ts, t_total)
    return log, Qbase


def _run_fault(cname, gains, objs, Q0, Ts, t_total, t_fault=1.0, fault_duration=0.15, Gfault=25.0):
    _, _, _, _, Qbase = init_plant_plc(objs, Q0)
    gains = dict(gains)
    if cname == 'LQI':
        gains['Qeq'] = Qbase
    qref = lambda t: Qbase
    fault_schedule = lambda t: (Gfault, 0.0) if (t_fault <= t < t_fault + fault_duration) else (0.0, 0.0)
    log, _, _ = run_plc_test(cname, gains, objs, Q0, qref, Ts, t_total, fault_schedule=fault_schedule)
    return log


def _run_voltage_chain(cname, gains, objs, Q0, Ts, schedule, seg, t_event0=1.0):
    _, Pmech0, Einf_base, Vref0, Qbase = init_plant_plc(objs, Q0)
    gains = dict(gains)
    if cname == 'LQI':
        gains['Qeq'] = Qbase
    full_sched = list(schedule) + [0.0]
    t_total = t_event0 + seg * len(full_sched)

    def einf_at(t):
        if t < t_event0:
            return Einf_base
        idx = min(int((t - t_event0) // seg), len(full_sched) - 1)
        return Einf_base * (1.0 + full_sched[idx])

    qref = lambda t: Qbase
    log, _, _ = run_plc_test(cname, gains, objs, Q0, qref, Ts, t_total, einf_schedule=einf_at)
    return log


def _run_full_battery(report, objs, Ts, Q0_OP, Ar, Br, Cr, K, L, Kp, Ki, battery_config):
    gains_pi = dict(Kp=Kp, Ki=Ki)
    gains_lqi = dict(K=K, L=L, Ar=Ar, Br=Br, Cr=Cr, Qeq=None)
    gains_relay = dict(dead_band=0.01, rate=0.002)

    step_results = {}
    points = battery_config['points']
    for label, Q0, step, ttot in points:
        report(f"Escalon: {label}...")
        per_ctrl = {}
        Qbase_this = None
        for cname, gains in [('PI', dict(gains_pi)), ('LQI', dict(gains_lqi)), ('Rele', dict(gains_relay))]:
            log, Qbase_this = _run_one(cname, gains, objs, Q0, step, Ts, ttot)
            per_ctrl[cname] = log
        step_results[label] = dict(data=per_ctrl, Qbase=Qbase_this, target=Qbase_this + step)

    report("Falla...")
    fc = battery_config['fault']
    fault_results = {}
    for cname, gains in [('PI', dict(gains_pi)), ('LQI', dict(gains_lqi)), ('Rele', dict(gains_relay))]:
        fault_results[cname] = _run_fault(cname, gains, objs, Q0_OP, Ts, t_total=fc['t_total'],
                                           t_fault=fc['t_fault'], fault_duration=fc['duration'],
                                           Gfault=fc['Gfault'])

    report("Perturbacion de red encadenada...")
    vc = battery_config['voltage']
    voltage_results = {}
    for cname, gains in [('PI', dict(gains_pi)), ('LQI', dict(gains_lqi)), ('Rele', dict(gains_relay))]:
        voltage_results[cname] = _run_voltage_chain(cname, gains, objs, Q0_OP, Ts,
                                                      schedule=vc['schedule'], seg=vc['seg'],
                                                      t_event0=vc['t_event0'])

    report("Bateria completa.")
    return dict(step=step_results, fault=fault_results, voltage=voltage_results)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
