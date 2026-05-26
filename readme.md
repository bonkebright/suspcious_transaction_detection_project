# Project Setup and Running the Streamlit Application (Windows)

This guide will walk you through setting up your local environment and running the Streamlit anomaly detection application on a Windows machine.

## 1. Install Python

If you don't already have Python installed, please download and install a stable version (e.g., Python 3.9 or newer) from the official Python website:

[Python for Windows](https://www.python.org/downloads/windows/)

**Important**: During the installation process, make sure to check the option that says "Add Python to PATH" or "Add Python to environment variables". This will allow you to run Python commands from any directory in your command prompt.

## 2. Create a Virtual Environment (Recommended)

Creating a virtual environment helps manage project dependencies by isolating them from your system's global Python packages. This prevents conflicts between different projects.

1.  **Open your Command Prompt (CMD) or PowerShell.**
2.  **Navigate to your project directory.** This is the folder where your `app.py` file is located. For example, if your project is on your Desktop:

    ```bash
    cd C:\Users\Pc\Desktop\ML Assignment 2\
    ```

3.  **Create a new virtual environment** named `venv` (you can choose another name if you prefer):

    ```bash
    python -m venv venv
    ```

4.  **Activate the virtual environment:**

    ```bash
    .\venv\Scripts\activate
    ```

    You'll know the virtual environment is active when you see `(venv)` at the beginning of your command prompt line.

## 3. Install Dependencies

Now, install all the necessary Python libraries within your activated virtual environment. It's crucial to install `scikit-learn==1.6.1` to match the version used in the Colab notebook. This will resolve the `AttributeError` related to `_RemainderColsList` you might have encountered.

Run the following commands one by one:

```bash
pip install streamlit pandas numpy
pip install scikit-learn==1.6.1
```

## 4. Run the Streamlit Application

Once all dependencies are installed, you can launch your Streamlit application.

With your virtual environment still activated, run:

```bash
streamlit run app.py
```

This command will:
*   Start the Streamlit server.
*   Open your default web browser to the application, usually at `http://localhost:8501`.

If the browser doesn't open automatically, copy and paste `http://localhost:8501` into your web browser's address bar.

Your Streamlit anomaly detection application should now be running locally!
