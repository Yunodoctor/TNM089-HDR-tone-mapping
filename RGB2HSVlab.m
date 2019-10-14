function [ HSV_lab_noise ] = RGB2HSVlab(noise)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

f=MFTsp(15,0.0847,500);

noise_HSV(:,:,1) = conv2(noise(:,:,1),f,'same');
noise_HSV(:,:,2) = conv2(noise(:,:,2),f,'same');
noise_HSV(:,:,3) = conv2(noise(:,:,3),f,'same');

noise_HSV_after(:,:,1) = (noise_HSV(:,:,1) > 0) .* noise_HSV(:,:,1);
noise_HSV_after(:,:,2) = (noise_HSV(:,:,2) > 0) .* noise_HSV(:,:,2);
noise_HSV_after(:,:,3) = (noise_HSV(:,:,3) > 0) .* noise_HSV(:,:,3);

HSV_lab_noise = rgb2lab(noise_HSV_after);

end

