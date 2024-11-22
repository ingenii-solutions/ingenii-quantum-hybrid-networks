import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks


class QuantumInspiredImageProcessor:
    description = "Applies quantum-inspired image filter.  "
    "For each pixel, this transfomation weighs the original pixel intensity   "
    "by a relative measure of pairwise intensity difference versus total neighborhood contribution. "
    class_parameters = {
        "mu": "(float) Steepness factor. Higher values of mu lead to a sharper curve, enhancing the discrimination between different intensity levels," 
        "whereas lower values yield a more gradual curve, providing smoother transitions.",
        "neighbor_size": "(int) Neighborhood size.",
        "percentile": "(int): Percentile to choose the cluster values w. Specifically, we designate w2 to correspond to the p percentile of the pixel intensity distribution,"
                         "achieving a balanced emphasis on the brighter areas within the image. Then, w3, .., wL are evenly spaced.",
        "max_iter": "(int) Maximum number of iterations for the transformation.",
        "threshold": "(float) Convergence threshold based on Mean Absolute Difference.",
        "L": "(int) Number of cluster levels."
            }
    required_parameters = {}
    optional_parameters = {
        "mu": "(float) Steepness factor. ",
        "neighbor_size": "(int) Neighborhood size.",
        "percentile": "(int): Percentile to choose the cluster values w. ", 
        "max_iter": "(int) Maximum number of iterations for the transformation.",
        "threshold": "(float) Convergence threshold based on Mean Absolute Difference.",
        "L": "(int) Number of cluster levels."
    }
    example_parameters = {
        "mu": 0.4 ,
        "neighbor_size": 3,
        "percentile": 50, 
        "max_iter": 10,
        "threshold": 1e-5,
        "L": 12
    }

    def __init__(self, mu=0.40, neighbor_size = 3, percentile=95, max_iter=10, threshold=0.0001, L=8):
        
        self.mu = mu
        self.neighbor_size = neighbor_size
        self.percentile = percentile
        self.max_iter = max_iter
        self.threshold = threshold
        self.L = L

    def check_convergence(self, w1, w2):
        '''
        Check convergence of the quantum-inspired model. If the Mean Absolute Difference of two consecutive matrices is smaller than the threshold, the method has converged.
            w1 (np.array): Previous image matrix.
            w2 (np.array): Current image matrix.
        returns:
            (bool): True if the model has converged.
            mae (float): Mean Absolute Difference.
        '''
        mae = np.mean(np.abs(w1- w2))
        if mae < self.threshold:
            print('Final Mean Absolute Difference: {:0.4f}'.format(mae))
            return True, mae
        return False, mae

    def sum_neighbours(self, image):
        '''
         Computes the sum of the neighbourhood of each pixel.
            image (np.array): input image
        returns:
            sum_neighbourhood (np.array): Sum of neighbouring pixels for each pixel.
        '''
        # 1. Calculate image shape
        ht,_ = image.shape
        # 2. Calculate necessary padding
        pad = int(np.ceil((np.ceil(ht/self.neighbor_size)*self.neighbor_size - ht)/2))
        if pad>0: # Pad image with 0s if the size of the image is not a multiple of the neighbouring size
            image = np.pad(image, (pad,pad), mode='constant', constant_values=(0,0))
        # 3. Create a window view of the image
        window_view = sliding_window_view(image, (self.neighbor_size, self.neighbor_size))
        # 4. Sum the elements of the view
        sum_neighbourhood = np.sum(window_view, axis=(2,3))
        return sum_neighbourhood

    def transformation(self, I, cluster):
        """
        Applies a quantum-inspired transformation.

        Parameters:
            I (np.array): Input image.
            cluster (np.array): Array containing the cluster values [w0, w1, ..., wn].

        Returns:
            f (np.array): Quantum-inspired image transformation (one iteration).
            alpha (np.array): Alpha values 1 - (I_{i+p,j+q} - I_{ij}).
        """
        I[I<0] = 0
        I[I>1] = 1
        ht, wt = I.shape[1], I.shape[0] 
        # 1. Calculate the sum of intensities of the 3x3 window for each pixel
        S = self.sum_neighbours(I)

        # Select cluster
        cluster = np.array(cluster)
        d = cluster.shape[0]
        # f is the quantum-inspired activation function f = sum( 1/(lamb + e^-mu(x-S)) ) 
        f = np.zeros((ht, wt))
        Alpha = np.zeros((ht, wt))
        idx = np.arange(d)
        for i in range(ht):
            for j in range(wt):
                p_vals = np.clip(np.arange(i - 1, i + 2), 0, ht - 1) # Indices [i-1, i, i+1] padding with 0s at the edges
                q_vals = np.clip(np.arange(j - 1, j + 2), 0, wt - 1) # Indices [j-1, j, j+1] padding with 0s at the edges
                # Find the index where I[i,j] >= cluster[k] and I[i,j] <= cluster[k + 1]
                if len(idx[I[i,j]>=cluster])==0:
                    print(I[i,j], cluster)
                k = min(idx[I[i,j]>=cluster][-1], d-2)
                if k+1 >=len(cluster):
                    lamb = S[i, j] / (1. - cluster[k])
                else:
                    lamb = S[i, j] / (cluster[k+1] - cluster[k])            
                sum_ = 0.
                alpha_ = 0.
                for p in p_vals:
                    for q in q_vals:
                        alpha = (1 - (I[p, q] - I[i, j]))
                        x = I[p, q] * np.cos((np.pi * 2) *(alpha - S[i, j]))
                        y = 1 / (lamb + np.exp(-self.mu * (x - S[i, j])))
                        sum_ += y
                        alpha_+=alpha
                #print('Lambda: ',lamb, 'x: ', x, 'S: ', S[i,j])
                f[i, j] = sum_
                Alpha[i, j] = alpha_

        return f, Alpha


    def select_image(self, mae_list, image_list):
        """
        Selects the output of the image transformation after multiple iterations.

        Parameters:
            mae_list (list): List of Mean Absolute Differences between iterations.
            image_list (list): List of image transformations.

        Returns:
            image (np.array): Final selected image.
            selected_peak (int): Iteration index.
        """
        # We select the image that has a peak in the mae
        mae_list = mae_list[1:]
        if len(mae_list)==1:
            return image_list[1], 1
        peaks, _ = find_peaks(mae_list, height=0)
        if len(peaks)==0:
            selected_peak = np.argmin(np.abs(np.diff(mae_list))/mae_list[1:]) + 1
        else:
            # If there are more than 1 peaks, we select the mean value
            selected_peak = int(np.mean(peaks))
            # For longer iterations, we add 1 to the output
            if selected_peak>5:
                selected_peak+= 1
        # Return the final selected image
        selected_peak = max(selected_peak,1)
        # If the images converge to a constant color, choose intermediate steps
        density, _ = np.histogram(image_list[selected_peak].flatten()/255, bins=20, density=False)
        density = density/np.sum(density)
        if np.max(density)>0.8:
            for i in range(len(image_list)-1, 0, -1):
                density, _ = np.histogram(image_list[i].flatten()/255, bins=20, density=False)
                density = density/np.sum(density)
                if np.max(density)<0.75:
                    selected_peak = i
                    return image_list[selected_peak], selected_peak
            selected_peak = 1
        return image_list[selected_peak], selected_peak


    def process(self, im, save=True, image_path='./filtered_image.png'):
        """
        Applies the quantum-inspired transformation until convergence.

        Parameters:
            im (np.array): Input image.
            save (bool): If True, saves the final image.
            image_path (str): Path to save the final image.

        Returns:
            out (np.array): Final processed image.
            mae_list (list): List of Mean Absolute Differences between iterations.
            image_list (list): List of image transformations.
        """
        wt, ht = im.shape[1], im.shape[0]  # Dimensions of the matrix    

        # Determine cluster values w
        min_val, max_val = np.percentile(im.flatten()/255, self.percentile), 1
        dw = (max_val - min_val)/(self.L-3)
        w = [0] + list(np.arange(min_val, max_val+dw, dw)) 
        
        transformed_image = im.copy().astype(float) / 255  # Initial image 
        #transformed_image *= (np.pi / 2) # Normalize initial image to [0, pi/2]

        w1 = np.zeros((ht, wt))
        mae_list = [0]
        image_list = [im]
        # Evolve Quantum inspired image until convergence
        for t in range(self.max_iter):  # Limit iterations 
            if t==0:
                cluster=w
            else:
                cluster = [0 ,0.16, 0.32, 0.48 ,0.64, 0.80, 0.96, 1]

            intermediate_matrix, _, = self.transformation(transformed_image, cluster)  
            transformed_image, w2 = self.transformation(intermediate_matrix, cluster)
            stop, mae = self.check_convergence(w2, w1)
            mae_list.append(mae)
            image_list.append((transformed_image * 255).astype(np.uint8))
            if stop:# Check for convergence
                break
            w1 = w2
        print('Number of iterations: ', t+1)
        out, selected_image=  self.select_image(mae_list, image_list)
        print('Selected image: ', selected_image)
        if save:
            Image.fromarray(out).save(image_path)

        return out, mae_list, image_list
