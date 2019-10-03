function M = tonemap(Enorm3, image)

L1 = rgb2gray(Enorm3);
L2 = log(L1);
sumL = sum(L2(:));

n = size(Enorm3,1)*size(Enorm3,2);
Lavg = exp((1/n)*sumL);

a = 0.11;
T = (a/Lavg)*L1;

Tmax2 = max(T(:))*max(T(:));
b = 1 + T/Tmax2;

% Ltone = T.*b/(1+T);
Ltone = zeros(size(image,1),size(image,2));


for p=1:size(image,1)
    for h=1:size(image,2)
        Ltone(p,h) = (T(p,h))/(1+T(p,h));
    end
end

<<<<<<< HEAD
M = zeros(682,1023);
for k1 = 1:682
    for k2 = 1:1023
=======
M = zeros(size(image,1),size(image,2));
for k1 = 1:size(image,1)
    for k2 = 1:size(image,2)
>>>>>>> b108c6e4b4f3276562829b224d252f75b3b62d1f
        M(k1,k2) = Ltone(k1,k2)/L1(k1,k2);
    end
end


end


