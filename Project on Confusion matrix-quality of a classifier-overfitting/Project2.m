%{
• Confusion matrix 
• Quality of a classifier
• Overfitting
%}

%Task1
%Confusion matrix is a valuable tool while evaluating the 
% quality of a classifier. 100 first points belonged to the class C1
% And remaining 100 points belonged to class C2. Class relation 
% of the remaining 12 points is unknown.

data=load('Data.txt');
d=data(1:200,:);
p=[2;2] %Calculation of bias term w0.
w=1/sqrt(2)*[1; -1]; %Direction part of our classifier.
w0=-w'*p
temp1=zeros(100,1); %Classification storage for first 100 sample points.
temp2=zeros(100,1); %Classification storage for next 100 sample points.
for i=1:100
temp1(i)=d(i,:)*w+w0; 
temp2(i)=d(i+100,:)*w+w0;
end

%Task1 and 2
CM=zeros(2,2);
f1=find(temp1>0); %Correct classifications from class C1.
f2=find(temp2<0); %Correct classifications from class C2.
CM(1,1)=length(f1); %Number of correct C1 cases.
CM(1,2)=100-CM(1,1); %Number of C1 cases misclassified in to C2.
CM(2,2)=length(f2); %Number of correct C2 cases.
CM(2,1)=100-CM(2,2); %Number of C2 cases misclassified into C1.
acc=(CM(1,1)+CM(2,2))/200; %Accurary. Correct classifications. 0.51.
sens=CM(1,1)/(CM(1,1)+CM(2,1)); %Sensivity. True C1’s out of all classifications into C1. 0.51.
spec=CM(2,2)/(CM(2,2)+CM(1,2)); %Specifity. True C2’s out of all classifications into C2. 0.51.
%T2
pe=(CM(1,2)+CM(2,1))/200; %Probability of an error is about 49%

%Task 3
function p=nprob(mu,s,a,b) %Function itself.
dx=a:0.01:b; %Interval for integration.
C=1/(sqrt(2*pi*s)); %Constant for pdf normalization.
pdf=C*exp(-1/(2*s)*(dx-mu).^2); %Calculation of pdf values.
p=cumsum(pdf*0.01); %Numerical estimation of integral.
p=p(length(dx))-p(1); %Substraction of starting point value.F(b)-F(a).
pr=nprob(0,1,-3,3) %Function call

%Task4
y1=ndist(0,1,-1,6)*0.3 %0.3*N(0,1)
y2=ndist(3,2,-1,6)*0.7 %0.7*N(3,2)
x=-1:0.01:6
x=x'
delta=abs(y1-y2); %Intersection of y1 and y2.
plot(delta)
[val pos]=min(delta) %Position for intersection.
b=x(pos); % x-value in the position of intersection.
E1=nprob(3,2,-1,b); %Area 1.
E2=nprob(0,1,b,6); %Area 2.
pe=0.3*A1+0.7*A2; %Probability of an error. About 0.13.
function pdf=ndist(mu,s,a,b) %Just pdfs.
dx=a:0.01:b;
C=1/(sqrt(2*pi*s));
pdf=C*exp(-1/(2*s)*(dx-mu).^2);

%Task5
X5=[ones(101,1) t t.^2 t.^3 t.^4 t.^5]; %Create observation matrix.
w5=inv(X5'*X5)*X5'*y; %Calculate model parameters.
yh5=X5*w5; %Predict values using calculated model.
plot(t,y)
hold on
plot(t,yh5)



