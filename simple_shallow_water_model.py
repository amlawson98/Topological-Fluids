import numpy as np # Standard
import matplotlib.pyplot as plt #Standard
from tqdm import tqdm #Great progress bar
import copy # Standard
from matplotlib.widgets import Slider #HElps with graphing and watchign evolution

#Simple array manipulation schemes
def reorder_axis(arr, ax):
    pos_part, neg_part = np.array_split(arr, 2, axis =ax)
    rejoined_arr = np.concatenate((neg_part, pos_part), axis = ax)
    return rejoined_arr

def reorder_axes(arr, axes_arr):
    if len(axes_arr)==0:
        return arr
    else:
        return reorder_axes(reorder_axis(arr, axes_arr[0]), axes_arr[1:])

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
#USEFUL FUNCTIONS FOR AVERAGING AND MAKING REASONABLE NOISE (NOT ALL USED)
def gaussian(x, y, sigma, mu1 = 0, mu2 = 0):
	return (1/ (sigma* np.sqrt(2*np.pi))) * np.exp(-0.5 * (np.sqrt((x - mu1)**2 + (y - mu2)**2)/ sigma)**2)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def two_sided_sigmoid(size, half_drop, a1, a2):
	x = np.arange(size)
	return sigmoid((x - 10*half_drop - a1)/half_drop) - sigmoid((x+10*half_drop - a2)/half_drop) 

#SIMPLE FINITE DIFFERENCE DERIVATIVES
def drv_x(arr):
	return (np.roll(arr, 1, axis = 1) - np.roll(arr, -1, axis = 1) )/(2*x_step)

def drv_y(arr):
	derivatived = (np.roll(arr, 1, axis = 0) -np.roll(arr, -1, axis = 0))/(2*y_step)
	return derivatived

#SETTING UP INITIAL CONDITIONS AND GRID PARAMETERS
x_step = 1
y_step = 1

grid_size = 400
xx, yy   = np.meshgrid(np.arange(grid_size),np.arange(grid_size))
gaussian_for_noise = gaussian(xx, yy, grid_size/10, grid_size/2, grid_size/2)
h_arr = np.zeros((grid_size, grid_size))
u_arr = np.zeros((grid_size, grid_size)) 
v_arr = np.zeros((grid_size, grid_size))
f = 1

def random_force(noise_scale):
	#RANDOM FORCE FOR AN ENTIRE GRID. CUTS OFF HGH FREQUENCIES OF NOISE THT COULD DISRUPT FINITE DIFFERENCE
	unfourier_noise = np.random.normal(scale = noise_scale, size = (grid_size, grid_size))
	unfourier_noise_shifted = reorder_axes(gaussian_for_noise*unfourier_noise, (0, 1))
	unfourier_noise_shifted = unfourier_noise_shifted
	ftn = np.real(np.fft.fftn(unfourier_noise_shifted))
	smoothed_noise = 0.5*ftn + (1/8)*(np.roll(ftn, 1, axis=0) + np.roll(ftn, -1, axis=0) + np.roll(ftn, 1, axis=1) + np.roll(ftn, -1, axis=1)) 
	return smoothed_noise


def update_arrs(h_arr, u_arr, v_arr, time_step, f):
	#EULER METHOD TIME STEPPING GIVEN BY LINEAR SW EQUATIONS WITH DAMPING AND A RANDOM FORCE ON h
	noise_scale = 1.4
	drag = 0.1
	h_arr_change = +random_force(1.4* (0.05/time_step)) - drag*h_arr - drv_x(u_arr) - drv_y(v_arr) 
	u_arr_change = -drag*u_arr  -drv_x(h_arr) + f*v_arr#+  np.random.normal(scale = noise_scale,size = (200, 200))
	v_arr_change = -drag*v_arr  -drv_y(h_arr) - f*u_arr#+  np.random.normal(scale = noise_scale,size = (200, 200))
	h_arr_new = time_step*h_arr_change + h_arr
	u_arr_new = u_arr + time_step*u_arr_change
	v_arr_new = v_arr + time_step*v_arr_change

	return h_arr_new, u_arr_new, v_arr_new

#Setup time
num_timesteps = 200
time_step = 0.05
time_arr = np.arange(num_timesteps)*time_step

#SIMPLE SHEME TO EULER STEP THROUGH AND GATHER DATA
h_total_array = [h_arr]
u_total_array = [u_arr]
v_total_array = [v_arr]
for i in tqdm(range(num_timesteps)):
	h_arr, u_arr, v_arr = update_arrs(h_arr, u_arr, v_arr, time_step, f)
	h_total_array.append(copy.copy(h_arr))
	u_total_array.append(copy.copy(u_arr))
	v_total_array.append(copy.copy(v_arr))

h_total_nparr = np.array(h_total_array)
u_total_nparr = np.array(u_total_array)
v_total_nparr = np.array(v_total_array)

#GROUP DATA TOGETHER FOR EASY SAVING/HANDLING
huv_nparr = np.stack((h_total_nparr, u_total_nparr, v_total_nparr), axis = 3)

#DYNAMIC PLOTTING OF DATA
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

pcolormesh = plt.pcolormesh(h_arr)
fig.colorbar(pcolormesh)
plt.set_cmap('bwr')
pcolormesh.set_clim(vmin=-0.3, vmax=0.3)
ax.margins(x=0)
ax.set_title('h(t, x, y) for f=%s, drag=%s, noise_magnitude=%s/Delta t, \nDelta x = Delta y = 1, Delta t = %s'%(f, 0.1, np.round(1.4*0.05, 2), time_step))
axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
time_index = 0

stime= Slider(axtime, 'time', time_arr[0], time_arr[-1],
 valinit=time_arr[time_index], valstep=time_arr[1]- time_arr[0])


def update(val):
    time = stime.val
    time_index = find_nearest_index(time_arr, time)
    new_array = np.array(h_total_array[time_index])
    pcolormesh.set_array(new_array.ravel())  
    fig.canvas.draw_idle()


stime.on_changed(update)
plt.show()
