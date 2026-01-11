import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import cv2
from scipy.ndimage import convolve1d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# model details
# initial conditions
a = 1
mu = 1

variance_init = 1.5  # Variance of initial distribution
sd_limit_init = 4  # How far we compute the initial configuration away in standard deviations
variance_kernel = 0.5  # Variance of the convolution kernel
sd_limit_kernel = 2  # How far we compute the kernel away in standard deviations
radius_kernel = sd_limit_kernel * variance_kernel  # Radius we compute the kernel up to
radius_init = sd_limit_init * variance_init  # Radius we compute the initial distribution up to



def get_typed_input(prompt, expected_type=int, allowed_values=None):
    while True:
        try:
            user_input = expected_type(input(prompt))
            if allowed_values is not None and user_input not in allowed_values:
                raise ValueError('Number outside of range')
            return user_input
        except ValueError as e:
            print(f"Error: {e}.")

def initialize_parameters():
    dimension = get_typed_input("Dimension of the simulation (1 or 2)?", allowed_values=[1, 2])
    T = get_typed_input("Total simulation time (recommended 20)?")
    L = get_typed_input("Simulation space size (recommended 25)?")
    dx = get_typed_input("dx parameter (recommended 0.1)?", expected_type=float)
    dt = get_typed_input("dt parameter (recommended 0.001)?", expected_type=float)
    k = get_typed_input("Number of divisions (recommended 10)?",expected_type=int)
    dirac = get_typed_input("Dirac convolution kernel (0 or 1)?", allowed_values=[0, 1])
    return dimension, T, L, dx, dt, k, dirac

def function(x, variance, dimension):
    return math.exp(-(sum(z ** 2 for z in x)) / (2 * variance)) / math.sqrt((2 * math.pi * variance) ** dimension)

def scale_function(f, dx):
    return lambda x: f([z * dx for z in x])

def convolution_matrix(g, size):
    return [g([i - size]) for i in range(2 * size + 1)]

class OneDimensional:
    f_init = lambda x: OneDimensional.function(x, variance_init)
    f_kernel = lambda x: OneDimensional.function(x, variance_kernel)
    def __init__(self, dx,dt,time=0,k=1, dirac=False, diameter_grid=None, initial_condition=None,centre=True):
        self.matrix = np.zeros((k, diameter_grid))
        self.divisions = k
        self.diameter_grid = diameter_grid
        self.dirac = dirac
        self.time = time
        self.dt=dt
        self.conv_matrix = OneDimensional.initialise_convolution()
        self.centre=centre
        self.shifted=0
        if initial_condition is not None:
            self.matrix = initial_condition
    @staticmethod
    def initialise_convolution():
        return np.array(convolution_matrix(scale_function(OneDimensional.f_kernel,dx),
                                           radius_kernel_scaled))
    @staticmethod
    def function(x, variance):
        return function(x,variance,1)
    def get_laplacian(self):
        Ztop = self.matrix[:, 0:-2]
        Zbottom = self.matrix[:, 2:]
        Zcenter = self.matrix[:, 1:-1]
        return Ztop + Zbottom - 2 * Zcenter

    def convolution(self):
        if not self.dirac:
            return convolve1d(self.matrix, weights=self.conv_matrix, mode='reflect')
        return np.array(self.matrix)
    def convolution_sum(self):
        if not self.dirac:
            return convolve1d(self.get_sum(), weights=self.conv_matrix, mode='reflect')
        return sum(self.matrix)
    def show_patterns(self, ax=None, axis_off=True):
        V = np.zeros(self.diameter_grid)
        for A in self.matrix:
            V += A
            ax.plot(V)
        ax.set_title(f'$t={self.time:.2f}$')
        if axis_off:
            ax.set_axis_off()
        return
    def get_sum(self):
        return np.array([sum([self.matrix[j, i] for j in range(self.divisions)])
                for i in range(self.diameter_grid)])
    def shift_left(self,k=1):
        if k<0:
            return
        self.shifted += k
        for i in range(self.divisions):
            inter=self.matrix[i,k:]
            self.matrix[i,:] = np.concatenate((inter, np.zeros(k)))
        return
    def update(self):
        t = self.time
        V = self.matrix
        dt=self.dt
        k_1 = self.f( V,t)
        k_2 = self.f(V + dt / 2 * k_1,t + dt / 2)
        k_3 = self.f( V + dt / 2 * k_2,t + dt / 2)
        k_4 = self.f( V + dt * k_3,t + dt)
        W = V + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        W[:, 0] = W[:, 1]
        W[:, -1] = W[:, -2]
        self.time += dt
        self.matrix = W

    def recentre(self):
        if self.centre:
            sum=self.get_sum()
            radius=self.diameter_grid//2
            k=-radius
            threshold=0.5*sum[0]
            while k+radius<len(sum) and sum[k+radius]>threshold:
                k+=1
            if k>=0:
                self.shift_left(k)
        return

    def f(self, V,t=-1):
        if t==-1:
            t=self.time
        conv = self.convolution_sum()
        big_conv=np.zeros(V.shape)
        for i in range(len(V[:, 0])):
            big_conv[i]=conv
        W = np.zeros(V.shape)
        deltaV = self.get_laplacian() / dx ** 2
        Vc = V[:, 1:-1]
        # Allen-Kahn equation
        #W[:, 1:-1] = a * deltaV + mu * Vc * (1 - big_conv[:,1:-1])*(2*big_conv[:,1:-1]-1)
        # Fisher-KPP equation
        W[:, 1:-1] = a * deltaV + mu * Vc * (1 - big_conv[:,1:-1])

        #boundary conditions
        W[:, 0] = V[:, 1]
        W[:, -1] = V[:, -2]
        return W

    @staticmethod
    def initialise_heaviside(min_point, max_point,diameter_grid):
        max_point = min(diameter_grid, max_point)  # truncate in case given wrong values
        min_point = max(0, min_point)
        V = np.zeros(diameter_grid)
        V[min_point:max_point] += np.ones(max_point - min_point)
        return V

# Similar modifications can be made to the TwoDimensional class

def main2(dx,dt,k, dirac,diameter_grid,iterations,fps=30,number_plots=9):
    U = TwoDimensional(dx,dt,k=k, dirac=dirac, diameter_grid=diameter_grid)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    step_plot = iterations // number_plots
    for i in range(iterations):
        if i % step_plot == 0 and i < number_plots * step_plot:
            ax = axes.flat[i // step_plot]
            U.show_patterns(ax=ax)
            ax.set_title(f'$t={i * dt:.2f}$')
        U.update()
    plt.show()

def main1(dx,dt,k, dirac,diameter_grid,iterations,fps=30,number_plots=30,centre=True):
    #initial conditions
    interval = int(diameter_grid / 2 / k)
    step_plot = iterations // number_plots
    U = OneDimensional(dx,dt,k=k, dirac=dirac, diameter_grid=diameter_grid,centre=centre)
    for i in range(k):
        U.matrix[i, :] = OneDimensional.initialise_heaviside(i * interval, (i + 1) * interval,
                                                             diameter_grid=diameter_grid)
    #video
    height, width = 480, 640  # Dimensions de la vidéo
    video = cv2.VideoWriter('video_collage.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    #Letting DE evolve
    times=np.zeros(0)
    shifted=np.zeros(0)
    for i in range(iterations):
        if i % step_plot == 0:
            fig,ax= plt.subplots()
            U.show_patterns(ax=ax)
            ax.set_title(f'$t={i * dt:.2f}$')
            #ax.xlabel('x-coordinate')
            #ax.ylabel('height')
            times=np.append(times,U.time)
            shifted=np.append(shifted,U.shifted*dx)
            U.recentre()
            canvas = FigureCanvas(fig)
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img)
            plt.close(fig)
        U.update()
    video.release()
    cv2.destroyAllWindows()

    #Show a plot of the wave speed over time
    plt.plot(times, shifted)
    plt.xlabel('Time')
    plt.ylabel('Shifted')
    plt.title('Shifted vs Time')
    plt.show()

if __name__ == "__main__":
    #dimension, T, L, dx, dt, k, dirac = initialize_parameters()
    dimension, T, L, dx, dt, k, dirac = 1, 20, 25, 0.1, 0.001, 10, 0
    fps=30
    # scaling
    radius_init_scaled = int(radius_init / dx)
    radius_kernel_scaled = int(radius_kernel / dx)
    diameter_kernel_scaled = 2 * radius_kernel_scaled + 1
    radius_grid = int(L / dx)  # size of 2d grid
    diameter_grid = 2 * radius_grid + 1  # since it goes both positive and negative
    iterations = int(T / dt)  # number of iterations

    # graphs
    number_plots = fps*T
    if dimension == 1:
        main1(dx=dx, dt=dt, k=k, dirac=dirac,diameter_grid=diameter_grid,iterations=
              iterations,fps=fps,number_plots=number_plots,centre=True)
    elif dimension == 2:
        main2()


class TwoDimensional:
    f_init = lambda x: TwoDimensional.function(x, variance_init)
    f_kernel = lambda x: TwoDimensional.function(x, variance_kernel)
    def __init__(self, dx,dt,time=0,k=1, dirac=False, diameter_grid=None, initial_condition=None):
        self.matrix = np.zeros((k, diameter_grid))
        self.divisions = k
        self.diameter_grid = diameter_grid
        self.dirac = dirac
        self.time = time
        self.dt=dt
        self.conv_matrix = TwoDimensional.initialise_convolution()
        if initial_condition is not None:
            self.matrix = initial_condition
    @staticmethod
    def initialise_convolution():
        return np.array(convolution_matrix(scale_function(TwoDimensional.f_kernel,dx),
                                           radius_kernel_scaled))
    @staticmethod
    def function(x, variance):
        return function(x,variance,2)
    def get_laplacian(self):
        Ztop = self.matrix[0:-2, 1:-1]
        Zbottom = self.matrix[2:, 1:-1]
        Zleft=self.matrix[1:-1, 0:-2]
        Zright=self.matrix[1:-1, 2:]
        Zcenter = self.matrix[1:-1, 1:-1]
        return (Ztop + Zbottom+Zright+Zleft) - 4 * Zcenter

    def convolution(self):
        if not self.dirac:
            return scipy.convolve2d(self.matrix, self.conv_matrix, mode='symm')
        return self.matrix
    def show_patterns(self, ax=None, axis_off=True):
        ax.imshow(self.matrix, cmap=plt.cm.copper,interpolation='bilinear',extent=[-1, 1, -1, 1])
        ax.set_title(f'$t={self.time:.2f}$')
        ax.set_axis_off()
        return ax
    def update(self):
        t = self.time
        V = self.matrix
        dt=self.dt
        k_1 = self.f( V,t)
        k_2 = self.f(V + dt / 2 * k_1,t + dt / 2)
        k_3 = self.f( V + dt / 2 * k_2,t + dt / 2)
        k_4 = self.f( V + dt * k_3,t + dt)
        W = V + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

        # boundary conditions
        W[0,:]=W[1,:]
        W[-1, :] = W[-1, :]
        for Z in W:
            Z[0] = Z[1]
            Z[-1] = Z[-2]
        self.time += dt
        self.matrix = W

    def f(self, V,t=-1):
        if t==-1:
            t=self.time
        conv = self.convolution()
        W = np.zeros(V.shape)
        deltaV = self.get_laplacian() / dx ** 2
        Vc = V[:, 1:-1]
        # Allen-Kahn equation
        W[1:-1, 1:-1] = a * deltaV + mu * Vc * (1 - conv[1:-1,1:-1])*(2*conv[1:-1,1:-1]-1)
        # Fisher-KPP equation
        #W[1:-1, 1:-1] = a * deltaV + mu * Vc * (1 - conv[1:-1,1:-1])


        return W

    @staticmethod
    def initialise_heaviside(min_point, max_point,diameter_grid):
        max_point = min(diameter_grid, max_point)  # truncate in case given wrong values
        min_point = max(0, min_point)
        V = np.zeros(diameter_grid)
        V[min_point:max_point] += np.ones(max_point - min_point)
        return V


class TwoDimensional1():

    def convolution_matrix(g, size):
        return [[g([i, j] - size * np.ones(2)) \
                 for i in range(2 * size + 1)] for j in range(2 * size + 1)]

        # This function intialises the initial condition

    def initialise_specific(radius_initial, g, zero_point=[radius_grid, radius_grid]):
        radius_initial = min(radius_grid, radius_initial)  # truncate in case given wrong values
        radius_initial_array = radius_initial * np.ones(2)
        diameter_initial = 2 * radius_initial + 1
        V = np.zeros(tuple(diameter_grid * np.ones(2, dtype=int)))
        centre = [[g([i, j] - radius_initial_array) for i in range(diameter_initial)] for j in range(diameter_initial)]
        V[zero_point[0] - radius_initial:zero_point[0] + radius_initial + 1, \
        zero_point[1] - radius_initial:zero_point[1] + radius_initial + 1] += centre
        return V
        # We use Runge Kutta 4 to solve the autonomous system of ODEs