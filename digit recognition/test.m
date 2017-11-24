function [] = test(image,X,Theta1,Theta2)

img=imread(image);
img=rgb2gray(img);
[ix,iy]=size(img);
img1=zeros(size(X,2),1);
img1(1:iy)=img(1,:);
for i=1:ix
  img1((i-1)*iy+1:i*iy)=img(i,:);
end
X(1,:)=img1(:);
pred = predict(Theta1, Theta2, X);
pred(1)


end

