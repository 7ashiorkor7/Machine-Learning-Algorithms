%Key ideas
% General idea of classification
% Naïve Bayes classifier using categorical data
% Recalling normal distribution
% Nearest neighbor algorithm
% Distance measures
% Linear discriminant classifier
% Naïve Bayes classifier using numerical data

C1=data3(1:100,:);
C2=data3(101:200,:);
points=data3(201:220,:);

points=points'
denom1=2*pi*sqrt(det(s1))
denom2=2*pi*sqrt(det(s2))
p1=zeros(20,1);
p2=zeros(20,1);
for i=1:20
    p1(i)=1/denom1*exp((-0.5*(points(:,i)-m1)'*inv(s1)*(points(:,i)-m1)));
    p2(i)=1/denom2*exp((-0.5*(points(:,i)-m2)'*inv(s2)*(points(:,i)-m2)));
end
stem(p1)
hold on
stem(p2,'g')
%p1 gives greater values for first ten samples, while p2 gives 
% greater values for remaining ten sample points.
% All samples are correctly Classified, so accuracy, sensivity and specifity are all 100%.


points=points';
m1=m1'
m2=m2'
D1=zeros(20,1);
D2=zeros(20,1);
for i=1:20
    x=points(i,:)
    D1(i)=sqrt((x-m1)*inv(s1)*(x-m1)');
    D2(i)=sqrt((x-m2)*inv(s2)*(x-m2)');
end
stem(D1)
hold on
stem(D2,'g')
%Mahalanobis Distances from all sample points to mean points m1 and m2.
% First ten points are closer to m1 , While remaining ten points are closer to m2


K1=zeros(20,1);
for i=1:20
    Y=data3(1:200,:);
    x=points(i,:);
    [idx d]=knnsearch(Y,x,'K',1,'Distance','euclidean');
    K1(i)=idx;
end
K3=zeros(20,3);
for i=1:20
    Y=data3(1:200,:);
    x=points(i,:);
    [idx d]=knnsearch(Y,x,'K',3,'Distance','euclidean');
    K3(i,:)=idx;
end
%Row numbers of nearest neighbors for all test samples.
%First 10 belong to class C1. If row number is less than 
% 101, the classification is correct. For remaining 10 points the row number of 
% nearest neighbor should be greater than or equal to 101
%If we use three nearest neighbors, one of Them comes from class C2 for two points that come from class C1 
% In this task there is no difference for classification If we use one or three nearest neighbors. However,
% If neighbors come from “wrong” class we should be
% careful in making decisions from such points


K1=zeros(20,3);
for i=1:20
    Y=data3(1:200,:);
    x=points(i,:);
    [idx d]=knnsearch(Y,x,'K',1,'Distance','cityblock');
    K1(i,1)=idx;
    [idx d]=knnsearch(Y,x,'K',1,'Distance','cosine');
    K1(i,2)=idx;
    [idx d]=knnsearch(Y,x,'K',1,'Distance','chebychev');
    K1(i,3)=idx;
end

n1=100;
n2=100;
C=(1/(n1+n2))*(n1*s1+n2*s2);
mu1=m1';
mu2=m2';
b=inv(C)*(mu1-mu2);
cl=zeros(20,1);
for i=1:20
    x=points(i,:)';
    y=b'*(x-0.5*(mu1-mu2));
    cl(i)=y;
end
Y1=repmat('C1',100,1);
Y2=repmat('C2',100,1);
Y=[Y1; Y2];
Y=cellstr(Y);
Y=categorical(Y);
LDM=fitcdiscr(data3(1:200,:),Y);
clldm=predict(LDM,data3(201:220,:));

NBM=fitcnb(data3(1:200,:),Y);
clnbm=predict(NBM,data3(201:220,:));





