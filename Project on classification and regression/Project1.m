%{
Outlining the purpose of any machine learning algorithm
    • Solving a problem
    • Classification (Predicting correct class for an unknown sample)
• Regression (Predicting variable value using available data)
• Outlining the importance of visual and geometrical intepretation
• Outlining the need of machine learning algorithms
%}

%{
1. Data.txt file contains 212 two dimensional points. Points from location 1 to location 100 belong to class C1, points from location 101 to location 200 belong to class C2. The origin of the remaining 12 points is unknown. They belong to either class C1 or C2. Points from class C1 and C2 are shown in fig.1. Find the unit weight vector w that is perpendicular to line l that passes through points p1=(- 2,6) and p2=(6,-2).
Figure 1. Points from class C1 are blue and points from class C2 are green.
2. Find such a threshold t for which wTx<t when x belongs to class C1 and wTx>t when x belongs to class C2. Use your threshold and classify remaining 12 points in file Data.txt to class C1 and C2. (Note wTx=w•x, where T is transpose operator).
3. Modify your classifier such that the classification can be done comparing the result to zero instead to that of t in previous task.
4. Calculate the projections of points from class C1 and C2 on the directions of w and l. Draw histograms for both directions and interpret your results.
5. Let us consider the points from class C1 alone. (from position 1 to position 100). The probability to an event that a point p1 is within some range from point p2 can be considered as a function of distance d(p1,p2) between the points. What is the probability for the event that a point in class C1 belong to the circular area with center point of (0,0) and the radius that is the mean of all distances of points in C1 from the center point (0,0)?
6. Fit a linear regression model to the data using points from 1 to 200. Inspect your result visually and consider what kind of problems you may encounter later if you use your model with new data.    

%}

%Task1
data=load('Data.txt')
C1=data(1:100,:);
C2=data(101:200,:);
p1=[-2;6] %Points
p2=[6;-2]
dl=p2-p1; %Direction of line l.
ll=sqrt(dl'*dl);%Length of direction vector dl.
dl=dl/ll; %Normalization to unit length.
%l and w perpendicular to each other if dot(w,dl)=0.
%w(1)*dl(1)+w(2)*dl(2)=0. 
%w(1)*dl(1)=-w(2)*dl(2)
%w(2)=-w(1)dl(1)/dl(2)
%Let w(1)=1/sqrt(2); This can also be randomly selected. 
w=[1/sqrt(2); 0]; %w(2) is unknown, but can be calculated using 
w(1).
w(2)=-w(1)*dl(1)/dl(2);
% %w=(1/sqrt(2), 1/sqrt(2))
% w=[1/sqrt(2);1/sqrt(2)]
dot(w,dl); %Check

%Task2
c1=zeros(100,1); %Store for lengths of projections in the direction of w from class c1.
c2=zeros(100,1); %Store for lengths of projections in the direction of w from class c2.
for i=1:100 %Points from class c1.
    p=data(i,:)
    t=w'*p' %Dot product. 
    c1(i)=t
end
for i=101:200 %Points from class c2.
    p=data(i,:)
    t=w'*p'
    c2(i-100)=t
end
[v1 p1]=max(c1) %Point from class c1 that is "fartest" from origin.
[v2 p2]=min(c2) %Point from class c2 that is "closest" to origin.
t=(v2+v1)/2 %Mean of minimum and maximum projections. Just one possible threshold (3.1)

res=zeros(12,1); %Store for projections of remaining 12 points.
for i=1:12
    p=data(i+200,:);
    v=w'*p'; %Dot product. w is originally column vector and p is row vector. Transposed here.
    res(i)=1; %Just a substitution.
if v>t %If length of the projection is greater that t, correct class is C2.
    res(i)=2;
end
end
%res'=[1 1 1 1 1 1 2 2 2 2 2 2]

%Task3
m=mean(data(1:200,:)) %Centering the data. Subtract mean from each case.
m=repmat(m,200,1)
centered=data(1:200,:)-m
c21=zeros(100,1); %Projections from c1.
c22=zeros(100,1); %Projections from c2.
for i=1:100
    p=centered(i,:)
    y=w'*p'
    c21(i)=y
end
for i=101:200
    p=centered(i,:)
    y=w'*p'
    c22(i-100)=y
end

%Task4
%T4 %Similar to task 1, but direction is d instead that of 
w.
c31=zeros(100,1);
c32=zeros(100,1);
for i=1:100
    p=data(i,:)
    t=dl'*p'
    c31(i)=t
end
for i=101:200
    p=data(i,:)
    t=dl'*p'
    c32(i-100)=t
end
ccw=[c1;c2] %Projecions in the direction of w.
ccl=[c31;c32] %Projecions in the direction of d.
hist(ccw)
figure
hist(ccl)

%Task5
rads=zeros(100,1) %Store for individual radii from class C1.
for i=1:100
    p=data(i,:);
    r=sqrt(p*p')
    rads(i)=r;
    end
m=mean(rads) %Mean of radii.
f=find(rads<m)
prob=length(f)/100 %About 0.58.

%Task6
X1=[ones(200,1) data(1:200,1)]; %Observation matrix 
augmented with ones
y=data(1:200,1); %y to be predicted.
b=inv(X1'*X1)*X1'*y; %Parameters for our model.
xmin=min(X1(:,2));
xmax=max(X1(:,2));
range=xmax-xmin; %range for predictors.
xvals=xmin:range/200:xmax;
xvals=xvals';
Xh=[ones(201,1) xvals];
yp=Xh*b; %Predicted values
plot(data(:,1),data(:,2),'.');
hold on
plot(xvals, yp, 'g')