import tkinter as tk
from tkinter import ttk
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Application(tk.Tk):

    def __init__(self):

        tk.Tk.__init__(self)
        self.screen_width = tk.Tk.winfo_screenwidth(self)
        self.screen_height = tk.Tk.winfo_screenheight(self)
        self.canvas_frame = tk.Frame(self, width=self.screen_width, height=self.screen_height)
        self.canvas_frame.pack(side="top", fill="both", expand=True)
        #self.canvas_frame.grid(row=0, column=0)


        self.canvas = tk.Canvas(self.canvas_frame, width=self.screen_width - 100, height=self.screen_height)
        self.canvas.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        self.scroll_size = 0
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.canvas.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar.config(command=self.canvas.yview)
        self.canvas_frame.grid_propagate(0)
        self.canvas.grid_propagate(0)



    def draw_figure(self, figure, location):
        figure_canvas_agg = FigureCanvasAgg(figure)
        figure_canvas_agg.draw()
        figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
        figure_w = int(figure_w)
        figure_h = int(figure_h)
        image = tk.PhotoImage(master=self.canvas, width=figure_w, height=figure_h)
        self.canvas.create_image(location[0] + figure_w/2, location[1] + figure_h/2, image=image)
        tkagg.blit(image, figure_canvas_agg.get_renderer()._renderer, colormode=2)

        return image

if __name__ == '__main__':
    app = Application()
    app.mainloop()