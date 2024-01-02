import os
import multiprocessing as mp
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

import io
import sys
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QLineEdit, QPushButton, QLabel, QVBoxLayout, QDialog
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain import PromptTemplate
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



OPENAI_API_KEY = "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"

#agent = create_csv_agent(
    #OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    #"C:/Users/tmanett/Desktop/ai/Mock_Data.csv",
    #verbose=True)

class Canvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = Figure(figsize=(5, 4), dpi=150), plt.subplots()
        super().__init__(fig)
        self.setParent(parent)

    def plot_graph(self, graph_data):
        self.ax.clear()
        self.ax.plot(graph_data['x'], graph_data['y'])
        self.ax.set(xlabel='x', ylabel='y', title='Graph')
        self.ax.grid()
        self.draw()
    
class GraphDialog(QDialog):
    def __init__(self, graph_data):
        super().__init__()
        self.setWindowTitle("Graph")

        self.canvas = Canvas(self)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_graph(graph_data)

    def plot_graph(self, graph_data):
        self.canvas.plot_graph(graph_data)

class ChatGPTInterface(QMainWindow):
    def __init__(self):
        super().__init__()

        if os.path.exists('conversation_history.txt'):
            self.load_conversation('conversation_history.txt')

        self.setWindowTitle("mlseGPT Interface")
        self.resize(800, 600)  # Set the initial size of the window

        # Create the text editor
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            background-color: #000000;
            color: #f6f6f8;
            font-size: 16px;
            padding: 20px;
            border: none;
            border-top: 1px solid #dddddd;
            font-family: Arial, sans-serif;
        """)

        # Create the input field
        self.input_field = QLineEdit()
        self.input_field.setStyleSheet("""
            background-color: #ffffff;
            color: #333333;
            font-size: 16px;
            padding: 10px;
            border: 1px solid #dddddd;
            font-family: Arial, sans-serif;
        """)

        # Create the send button
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("""
            background-color: #4CAF50;
            color: #ffffff;
            font-size: 16px;
            padding: 10px;
            border: none;
            cursor: pointer;
        """)
        self.send_button.clicked.connect(self.send_question)

        # Create the exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("""
            background-color: #f44336;
            color: #ffffff;
            font-size: 16px;
            padding: 10px;
            border: none;
            cursor: pointer;
        """)
        self.exit_button.clicked.connect(self.close)

        # Create the clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet("""
            background-color: #f44336;
            color: #ffffff;
            font-size: 12px;
            padding: 5px;
            border: none;
            cursor: pointer;
        """)
        self.clear_button.clicked.connect(self.clear_chat)

        # Create the layout and set the central widget
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.input_field)
        layout.addWidget(self.send_button)
        layout.addWidget(self.clear_button)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Create a widget for the buttons layout
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout()
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.exit_button)
        buttons_widget.setLayout(buttons_layout)
        layout.addWidget(buttons_widget, alignment=Qt.AlignTop | Qt.AlignRight)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Create the language model agent
        self.agent = create_csv_agent(
            OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
            "C:/Users/tmanett/Desktop/ai/Mock_Data.csv",
            verbose=True
        )

        # Conversation history
        self.conversation = []

    def append_message(self, message, is_user=False):
        if isinstance(message, dict) and 'graph' in message:
            graph_data = message['graph']

                # Display the graph in a separate dialog window
            dialog = GraphDialog(graph_data)
            dialog.exec_()

        else:
            # Format the regular message with the appropriate prefix
            if is_user:
                formatted_message = f"<p>User: {message}<br><br>"
            else:
                formatted_message = f"<p>MLSE GPT: {message}<br><br>"

            # Append the formatted message to the text editor
            message_class = "user-message" if is_user else "model-message"
            html_message = f'<div class="{message_class}">{formatted_message}</div>'
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.text_edit.setTextCursor(cursor)
            self.text_edit.insertHtml(html_message)
            self.text_edit.ensureCursorVisible()
            QApplication.processEvents()  # Force GUI update

            # Move the cursor to the beginning of the text editor
            cursor.movePosition(QTextCursor.Start)
            self.text_edit.setTextCursor(cursor)

    def send_question(self):
        # Get the user's question from the input field
        question = self.input_field.text()
        self.input_field.clear()

        # Display the user's question in the GUI
        self.append_message(question, is_user=True)

        # Add the user's question to the conversation history
        self.conversation.append(question)

        # Use QTimer to delay the retrieval of the model's response
        QTimer.singleShot(0, self.retrieve_model_response)

    def retrieve_model_response(self):
        # Interact with the model using the conversation history
        response = self.agent.run(self.conversation)

        # Add the model's response to the conversation history
        self.conversation.append(response)

       # Check if the response contains graph data
        if isinstance(response, dict) and 'graph' in response:
            graph_data = response['graph']

        # Display the graph in a separate dialog window
            dialog = GraphDialog(graph_data)
            dialog.exec_()

        else:
        # Display the model's response in the GUI
            self.append_message(response)

    def clear_chat(self):
        # Clear the text editor
        self.text_edit.clear()

        # Clear the conversation history
        self.conversation.clear()

    def closeEvent(self, event):
        event.accept()
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication([])

    gui = ChatGPTInterface()
    gui.show()

    sys.exit(app.exec_())