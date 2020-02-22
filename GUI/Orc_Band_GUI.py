import sys
# sys.path.append('../figures')
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QTextEdit, QTextBrowser, QMessageBox, QFrame,
                             QGridLayout, QToolTip, QPushButton, QApplication, QProgressBar)
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import QCoreApplication, QSize, QThread, pyqtSignal
import time

import numpy as np
import pandas as pd
import pickle

from sklearn import linear_model
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from multiprocessing import freeze_support
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors


def load_model():
    """
    load the trained prediction model
    """
    with open('../model/regressor.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


class Runthread(QThread):
    """"
    The class is used to calculate and produce the prediction result
    """
    _signal = pyqtSignal(str)

    def __init__(self, sms,clear):
        super(Runthread, self).__init__()
        self.sms = sms
        self.clear = clear

    def __del__(self):
        self.wait()

    def run(self):
        self.clear
        sms = self.sms
        freeze_support()
        mols = Chem.MolFromSmiles(sms)  # transform smiles string to molecular structure
        if mols is None:
            raise TypeError('Invalid Smiles String')
        else:
            m = [Chem.MolFromSmiles(sms)]
            calc = Calculator(descriptors)
            raw_data = calc.pandas(m)  # calculate descriptors
            new = {'AXp-0d': raw_data['AXp-0d'].values,
                   'AXp-1d': raw_data['AXp-1d'].values,
                   'AXp-2d': raw_data['AXp-2d'].values,
                   'ETA_eta_L': raw_data['ETA_eta_L'].values,
                   'ETA_epsilon_3': raw_data['ETA_epsilon_3'].values}  # extract the five most useful descriptors data
            new_data = pd.DataFrame(index=[1], data=new)
            regressor2 = load_model()
            bg = regressor2.predict(new_data)[0]
        self._signal.emit(str(bg))


class Prediction(QWidget):
    def __init__(self):
        super(Prediction, self).__init__()

        self.setWindowIcon(QIcon('../docs/icon.ico'))


        self.input = QLineEdit(self)
        self.input.move(10, 65)


        self.reactant = QLabel('Smiles String:', self)
        self.reactant.move(10, 40)

        self.result = QLabel('Bandgap Prediction:', self)
        self.result.move(160, 40)


        self.resize(400, 360)
        self.setWindowTitle("OrcBand")

        self.plabel = QLabel(self)
        self.plabel.setFixedSize(200, 200)
        self.plabel.move(160, 60)


        self.outbg = QTextBrowser(self)
        self.outbg.setFixedSize(200, 30)
        self.outbg.move(160, 270)

        self.plabel.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:YaHei;}"
                                  )


        btn = QPushButton(self)
        btn.setText("Predict")
        btn.move(10, 150)
        btn.clicked.connect(self.start_login)
        btn.clicked.connect(self.show_molecule)

        cbtn = QPushButton(self)
        cbtn.setText("Clear")
        cbtn.move(10, 200)
        cbtn.clicked.connect(self.clear_input)
        cbtn.clicked.connect(self.clear_output)

        qbtn = QPushButton(self)
        qbtn.setText("Quit")
        qbtn.move(10, 250)
        qbtn.clicked.connect(QCoreApplication.instance().quit)

        QToolTip.setFont(QFont('SansSerif', 10))
        btn.setToolTip('<b>Predict the Bandgap</b> of input reactant')
        cbtn.setToolTip('<b>Clear</b> the contents on the interface')
        qbtn.setToolTip('<b>Exit</b> the execution instantly')


    def get_smi(self):
        """
        Accuqire the smile string
        """
        smi = self.input.text()
        return smi

    def clear_input(self):
        self.input.clear()

    def clear_output(self):  # problem
        """"
        clear the output
        """
        self.plabel.clear()
        self.outbg.clear()

    def show_molecule(self):
        smi = self.input.text()
        mol = Chem.MolFromSmiles(smi)
        Draw.MolToFile(mol, 'molecule.png', size=(200, 200))
        jpg = QPixmap('molecule.png').scaled(self.plabel.width(), self.plabel.height())
        self.plabel.setPixmap(jpg)

    def start_login(self):
        self.thread = Runthread(sms=self.get_smi(),clear = self.clear_output())
        self.thread._signal.connect(self.show_bandgap)
        self.thread.start()


    def show_bandgap(self, bgp):
        self.outbg.append(bgp)


    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to exit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = Prediction()
    my.show()
    sys.exit(app.exec_())


