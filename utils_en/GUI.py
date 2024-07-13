import tkinter as tk
from tkinter import messagebox
from utils_en.predict_review import *

def GUI():

    def click_submit():
        # This function is called when the user clicks the Submit button
        review = text.get("1.0", tk.END)
        # Get the text content entered by the user, capturing everything from the start to the end

        result = predict_input(model, word_to_index, review)
        # Call the predict_input function to get the result

        messagebox.showinfo("Sentiment Result", result)
        # Display the result in a popup window

    model, word_to_index = load_model()
    # Load the model and vocabulary

    root = tk.Tk()
    root.title("IMDb Analysis")
    root.geometry("300x250+575+300")
    # Create the main window and set its title, size, and position

    text_label = tk.Label(root, text="Input your review:")
    text_label.pack(pady=10)
    # Create a label to prompt the user for input, parent is the main window root
    # Set the vertical padding to 10 pixels
    # You can also directly add .pack after Label(root, text="Input your review:")
    # But it's not recommended because it will return 'None', making it difficult
    # to handle and call this widget later. So handling it separately like this is highly recommended
    # and looks more organized

    text = tk.Text(root, width=50, height=10)
    text.pack(padx=20, pady=10)
    # Create a multi-line text input box with a width of 50 and height of 10
    # Set the horizontal and vertical padding to 20 and 10 respectively

    submit_button = tk.Button(root, text="Submit", command=click_submit)
    submit_button.pack(pady=10)
    # Create a button that calls the click_submit function when clicked

    root.mainloop()
    # Start the Tkinter main event loop, this window starts running and waits for interaction
