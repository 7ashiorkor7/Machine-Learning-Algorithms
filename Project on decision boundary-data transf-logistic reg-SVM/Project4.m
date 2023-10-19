%{
Key ideas
• Decision boundary
• Data transformation
• Presentation of new algorithms
• Logistic regression
• Support vector machine
%}

m1=[3;6]; %Parameters for distributions.
s1=[0.5 0; 0 2];
m2=[3; -2];
s2=[2 0; 0 2];
C1=1/(2*pi*det(s1)) %Convinience substitution.
C2=1/(2*pi*det(s2))

xr=-6:0.1:10; % Values for x on the grid.
yr=xr; %Values for y on the grid
pdf1=zeros(length(yr),length(xr)); %Pdf values for our first normal distribution.
pdf2=zeros(length(yr),length(xr)); %Pdf values for our another normal distribution.
DecB1=zeros(length(y),length(x)); %calculated decision area for pdf1.
DecB2=zeros(length(y),length(x)); %calculated decision area for pdf2.
for row=1:length(yr) %Let us consider grid rows as y-coordinates
    for column=1:length(xr) %and columns as x-coordinates.
        p=[xr(column);yr(row)];
        v1=1/C1*exp(-0.5*(p-m1)'*inv(s1)*(p-m1));
        v2=1/C2*exp(-0.5*(p-m2)'*inv(s2)*(p-m2));
        pdf1(row,column)=v1;
        pdf2(row,column)=v2; 
    end
end

DB1=pdf1-pdf2; %Decide N(mu1,S1)
f1=find(DB1>0)
DB2=pdf2-pdf1;
f2=find(DB2>0) %Decide N(mu2,S2)


%Fitting multinomial logistic regression model on Iris data.
[coeff,dev,stats] = mnrfit(iris(:,1:4),categorical(iris(:,5))); 
%Points to be classified.
c1=[6.5 2.9 5.5 2.0];
c2=[5.0 3.4 1.5 0.2];
c3=[5.9 2.7 4.3 1.3];
c=[c1; c2; c3];
pr=mnrval(coeff,c) %Prediction of species using fitted model.


r11=coeff(1,1)+coeff(2:end,1)'*c1'; %ln(c0/c2) for first point.
r12=coeff(1,2)+coeff(2:end,2)'*c1'; %ln(c1/c2) for first point.
r21=coeff(1,1)+coeff(2:end,1)'*c2'; %ln(c0/c2) for second point.
r22=coeff(1,2)+coeff(2:end,2)'*c2'; %ln(c1/c2) for second point.
r31=coeff(1,1)+coeff(2:end,1)'*c3'; %ln(c0/c2) for third point.
r32=coeff(1,2)+coeff(2:end,2)'*c3'; %ln(c1/c2) for third point.
pc10=exp(r11)/(1+exp(r11)+exp(r12)) %calculation of probability of class 0 for first point.
pc11=exp(r12)/(1+exp(r11)+exp(r12)) %calculation of probability of class 1 for first point.
pc12=1/(1+exp(r11)+exp(r12)) %calculation of probability of class 2 for first point.
pc20=exp(r21)/(1+exp(r21)+exp(r22)) 
pc21=exp(r22)/(1+exp(r21)+exp(r22))
pc22=1/(1+exp(r21)+exp(r22))
pc30=exp(r31)/(1+exp(r31)+exp(r32))
pc31=exp(r32)/(1+exp(r31)+exp(r32))
pc32=1/(1+exp(r31)+exp(r32))


X=data3(1:200,:) %Datapoints for creating (training) SVM.
lbls1=ones(100,1) 
lbls2=ones(100,1)*2
y=[lbls1;lbls2]; %Class labels for training points.
mysvm=fitcsvm(X,y) %Training an SVM.
sv = mysvm.SupportVectors; % Support vectors.
w0=mysvm.Bias %Bias term for classifier.
w=mysvm.Beta %Weight vector w. %Weight vector for classifier.
%SVM classifier is now w(x)=w'*x+w0
l=sqrt(w'*w)
margin=2/l; %0.93


Iris=load('Irisbak.txt');
X1=Iris(1:40,2:5) %Datapoints for creating (training) SVM.
X2=Iris(51:90,2:5)
lbls1=Iris(1:40,6); 
lbls2=Iris(51:90,6);
y=[lbls1;lbls2]; %Class labels for training points.
X=[X1;X2]
irissvm=fitcsvm(X,y) %Training an SVM.
testdata1=Iris(41:50,2:5);
testdata2=Iris(91:100,2:5);
testdata=[testdata1; testdata2];
classlabels = predict(irissvm,testdata);


X3=Iris(101:140,2:5);
lbls3=Iris(101:140,6);
X=[X; X3];
y=[y;lbls3];
testdata3=Iris(141:150,2:5);
testdata=[testdata; testdata3];
SVMModel = fitcecoc(X,y);
classlabels=predict(SVMModel,testdata);
