function [delta_met] = lab2deltaE(LAB_ref, LAB_est)
%A function to calculate the mean and max of Delta E between the XYZ ref
%values and the estimated values

deltaE = (LAB_ref(:,:,1:3) - LAB_est(:,:,1:3)).^2;

deltaE_sqrt = sqrt(deltaE(:,:,1) + deltaE(:,:,2) + deltaE(:,:,3));

delta_met = mean(mean(deltaE_sqrt));
end