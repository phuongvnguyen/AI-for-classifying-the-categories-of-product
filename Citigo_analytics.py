#!/usr/bin/env python
# coding: utf-8

# $$\Large \color{green}{\textbf{Phân tích dữ liệu của Citigo, Kiotviet}}$$
# 
# 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# 
# Chương trình phân tích dữ liệu này thưc hiện bởi Nguyễn Văn Phương, dựa trên nền tảng $\textbf{Anacoda 1.9.7}$ và $\textbf{Python 3.7}$.
# 
# 
# 
# Toàn bộ Mã chương trình, bao gồm (.py,.ipynb, .html), có thể tải tại Kho trên trang Github của tôi theo đường dẫn dưới đây
# 
# https://github.com/phuongvnguyen/Kiotviet-Data-Analytics
# 
# Nếu các bạn chỉ cần xem các bước Nội dung phân tích (Tatble of Contents) và các Mã lệnh đã thực hiện thì chỉ cần mở tập tin Citigo_analytics.html. Khi các bạn ở tập tin định dạng .html này ra, ở phía bên trái của của Sổ chương trình sẽ có Bảng Nội dung (Tatble of Contents), để các bạn dễ dàng dịch chuyển tới các nội dung và đoạn mã mà bạn cần xem.
# 
# Nếu các bạn muốn thực hiện lại toàn bộ quá trình thì sử dụng tập tin Citigo_analytics.py hoặc Citigo_analytics.ipynb, đặc biệt khi sử dụng tập tin Citigo_analytics.ipynb, trên thiết bị của bạn nên cài đặt chương trình $\textbf{Jupyter Notebook Extension}$. Đây là chương trình vô cùng thuận tiện cho việc viết các câu lênh và phân tích.
# 
# 
# Để cài đặt chương trình này,các bạn thảm khảo tại :
# 
# https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html
# 
# Về Vai trò tác dụng của nó các bạn tham khảo tại:
# 
# https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231
# 
# $$\color{green}{\underline{\textbf{Quy trình chung cho đào tạo một chương trình Máy Học}}}$$
# 
# Các bước tiến hành để đào tạo một chương trình Máy Học như sau:
# 
# 1. Gọi các Thư Viện cần thiết liên quan tới xử lý dữ liệu thô, thuật toán...
# 
# 2. Nhập dữ liệu vào chương trình, có thể từ tập tin excel, hoặc csv, etc.
# 
# 3. Khai phá dữ liệu thô: $\color{red}{\underline{\textbf{bước quan trọng nhất và tốn nhiều thời gian nhất}}}$.
# 
# 4. Chuẩn bị dữ liệu cho đào tạo một chương trình Máy học, như tạo biến dummy, đồng bộ hoá phân phối và đơn vị đo lường của các biên, etc 
# 
# 5. Chọn biến, gồm có chọn các biến có liên quan, bỏ các biến không có ý nghĩa giải thích, hoặc làm giảm bớt kích cỡ dữ liêụ, etc.
# 
# 6. Đào tạo chương trình Máy Học, chọn thuật toán phù hợp nhất với dữ liệu, chọn hệ số tuỳ biến tối ưu cho thuật toán (hyperparameter tunning)
# 
# 7. Lưu chương trình.
# 
# $$\color{green}{\underline{\textbf{Kết quả phân tích}}}$$
# 
# Kết quả phân tích dữ liệu dựa trên cả hai phương pháp là Phân tích thông kê mô tả và Đồ thị (dạng cột, dạng Pie, và dạng Donut). Đặc biệt, Phương pháp phân tích bằng Đồ thị dạng Donut, cho chúng ta được cách nhìn tổng quan toàn bộ phân phối và thị phần của các nhóm hàng. Chi tiết các bạn có thể xem các Đồ thị này và các đoạn Câu lệnh trong chương trình. Một vài điểm phân tích nổi bật được đưa ra như sau.
# 
# Bộ Dữ liệu thô, thống kê gần 13,000 Mặt hàng đều thuộc về Nhóm hàng $\textbf{Sức khoẻ - làm đẹp}$ (phân theo tiêu chí Level 1). Đồ thị dạng Donut cho chúng ta thấy, trong Nhóm hàng Sức khoẻ - làm Đẹp này, lại được phân loại thành 6 Nhóm hàng nhỏ tiếp theo (phân loại theo tiêu chí Level 2), bao gồm:
# 
# 1. $\textbf{Chăm sóc da}$: có số lượng Mặt hàng nhiều nhất, với hơn 6,000 Mặt hàng, chiếm gần 47% Thị phần toàn Mặt hàng. Đồng thời, hơn 6,000 Mặt hàng Chăm sóc da, tiếp tục được phân loại vào 8 Nhóm Mặt hàng (phân theo Level 3). Ví dụ, nhóm hàng $\textbf{Tinh chất dưỡng ẩm & trắng, chống lão hoá}$, có tới hơn 2,500 Mặt hàng, chiếm hơn 19% thị phần, cao nhất trong toàn Mặt hàng.
# 
# 2. $\textbf{Trang điểm}$: Có số lượng Mặt hàng nhiều thứ hai, với gân 2,700 Mặt hàng, chiếm gần 21% Thị phần toàn Mặt hàng. Thêm nữa, gần 2,700 Mặt hàng Trang điểm này, lại được phân vào 7 Nhóm hàng nhỏ tiếp theo (phân loại theo Level 3), như $\textbf{Trang điểm môi}$ (862 Mặt hàng:6.62%), $\textbf{Trang điểm mắt}$ (533 Mặt hàng:4.09%), $\textbf{Trang điểm mắt}$ (428 Mặt hàng: 3.29%), etc 
# 
# $\textbf{}$
# 3. $\textbf{Chăm sóc cơ thể}$: Có số lượng Mặt hàng nhiều thứ 3, hơn 1,900 Mặt hàng, chiếm gần 15% Thị phần toàn Mặt hàng. Đồng thời nhóm hàng Chăm sóc cơ thể này, lại được chia làm 6 nhóm hàng nhỏ (phân loại theo Level 3), như $\textbf{Sữa tắm, xà bông, xà bông, tẩy tế bào chết cơ thể}$ (642 Mặt hàng: 4.93%), $\textbf{Sản phẩm khử mùi}$ (300 Mặt hàng: 2.3%), etc.
# 
# 4. $\textbf{Chăm sóc da tóc và da dầu}$: Có số lượng Mặt hàng nhiều thứ 4, với gần 1,700 Mặt hàng, chiếm gần 13% thị phần toàn Mặt hàng. Đồng thời, gần 1,700 Mặt hàng Chăm sóc da tóc và da dầu, lại tiếp tục được phân loại vào 7 Nhóm mặt hàng (theo tiêu chí phân loại Level 3), chẳng hạn như $\textbf{Dầu gội, dầu xả}$ (920 Mặt hàng: 7.06%), $\textbf{Dưỡng tóc, ủ tóc}$ (190 Mặt hàng: 1.46%), etc.
# 
# 5. $\textbf{Nước hoa}$: Có số lượng Mặt hàng ít thứ hai, với 405 Mặt hàng, chiếm  3.12% thị phần toàn Mặt hàng, bao gồm ba nhóm nước hoa, như $\textbf{Nước hoa nữ}$ (38 Mặt hàng: 1.77%), $\textbf{Nước hoa nam}$ (87 Mặt hàng: 0.67%), và $\textbf{Nước hoa khác}$ (86 Mặt hàng: 0.66%).
# 
# 6. $\textbf{Tinh dầu spa}$: Có số lượng Mặt hàng ít nhất, 217 Mặt hàng, chiếm chưa đầy 2% Thị phần toàn Mặt hàng.
# 
# Mặt, khác Đồ thị dạng cột cho chúng ta thấy, theo phân loại nhóm hàng tiêu chí Level 3, thì Nhóm hàng có số mặt hàng nhiều nhất thuộc về $\textbf{Tinh chất dưỡng ẩm & trắng, chống lão hoá}$, có tới hơn 2,500 Mặt hàng, chiếm hơn 19% thị phần. Tiếp theo là Nhóm hàng $\textbf{Dầu gội, dầu xả}$ (920 Mặt hàng: 7.1%), $\textbf{Sản phẩm chăm sóc da khác}$ (896 Mặt hàng: 6.91%), etc. Cuối cùng, nhóm mặt hàng có ít số lượng sản phẩm nhất, thuộc về Nước hoa, tinh dầu spa khác, và trang điểm mắt, có chưa tới 10 Mặt hàng, thị phần chưa đầy 1% toàn Mặt hàng.
# 
# Sau đây là quy trinh thực hiện phân tích chi tiết
# 
# 
# 
# 
# # Gọi các thư viện thuật toán cần thiết
# 
# Thông thường trong bước này mình viết một chương trình python ở một tập tin riêng, sau đó gọi hàm này áp dụng cho tất cả các trường hợp của dữ liệu. Tuy nhiên, vì trong trường hợp này, mình muốn chỉ ra những thuật toán nào thường hay sẽ sử dụng trong viết chương trình phân tích dữ liệu và Máy học nên mình tích hợp bước 1 này vào trong chương trình. Nếu các bạn quan tâm có thể tham khảo chương trình được viết riêng cho bước này ở Github của mình tại
# 
# ## Cho xư lý và phân tích dữ liệu thô
# 

# In[1]:


import os
import numpy as np
import pandas as pd
from scipy import stats
from pandas import set_option
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from feature_selector import FeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# ## Các thuật toán Máy Học
# ### Thuật toán tuyến tính

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import PassiveAggressiveRegressor


# ### Thuật toán phi tuyến tính

# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR


# ### Thuật toán cao cấp khác

# In[4]:


from sklearn.pipeline import Pipeline


# #### Các Phương pháp boosting
# 
# Hãy chắc chắn rằng thiết bị của bạn đã cài đặt gói xgboost, nếu không các bạn có thể bỏ qua nó

# In[5]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
#from xgboost import XGBClassifier
#from xgboost import XGBRegressor


# #### Các phương pháp bagging

# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor


# #### Kêt hợp các phương pháp

# In[7]:


from sklearn.ensemble import VotingClassifier


# ## Phương pháp đánh giá mô hình

# In[8]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error


# ##  Định nghia biến cho in kết quá

# In[9]:


Purple= '\033[95m'
Cyan= '\033[96m'
Darkcyan= '\033[36m'
Blue = '\033[94m'
Green = '\033[92m'
Yellow = '\033[93m'
Red = '\033[91m'
Bold = "\033[1m"
Reset = "\033[0;0m"
Underline= '\033[4m'
End = '\033[0m'


# # Nhập dự liệu
# 
# Trước khi nhập dữ liệu, bạn chắc chắn rằng thư mục làm việc của bạn có chứa tập tin bạn cần nhập

# In[10]:


print(Bold + Blue + 'Thư mục làm việc hiện tại của bạn:' + End)
print(os.getcwd())


# In[11]:


citigo_data=pd.read_csv('Product Classify - Rawdata.csv')


# # Khai phá dữ liệu thô
# 
# Có hai phương pháp khai phá dữ liệu thô
# 
# 1. Phương pháp thống kê mô tả.
# 
# 2. Phương pháp mô tả bằng đồ thị.
# 
# ## Phương pháp thống kê mô tả
# ### Chọn vài dữ liệu mẫu
# Bạn có thể chọn xem 10, 20, hoặc nhiều hơn số quan sát đầu tiên. Ơ đây mình muốn xem 10 quan sát đầu tiên

# In[12]:


citigo_data.head(10)


# ### Kiểm tra kích cỡ dữ liệu

# In[13]:


print('--------------------')
print(Bold+ Red  + 'Kích cỡ dữ liệu thô:' + End)
print('--------------------')
print(Bold + Blue + 'Số quan sát: {}'.format(len(citigo_data)))
print(Bold + Blue + 'Số cột: {}'.format(len(citigo_data.columns)))
print(Bold+ Blue + 'Danh sách cột:' + End)
print(citigo_data.columns)
print('--------------------')


# ### Kiểm tra thiếu dữ liệu

# In[14]:


print(Bold + Darkcyan +'Số dữ liêu bị thiếu:'+ End)
print(citigo_data.isnull().sum())


# $$\textbf{Nhận xét:}$$
# 
# Từ kết quả in ra ở trên cho thấy rất may mắn là dữ liệu thô của chúng ta không có cột nào bị thiếu dữ liệu.

# ### Kiểm tra dạng dữ liệu

# In[15]:


print(Bold + Green+  'Dạng dữ liệu:'+ End)
print(citigo_data.dtypes)


# $$\textbf{Nhận xét:}$$
# Toàn bộ dữ liệu định dạng là đối tượng (object) không phải dạng số (numerical). Từ đó điều đầu tiên chúng ta nghĩ tới là sẽ sư dụng các thuật toán liên quan tới dạng này cho đào tạo Máy Học (nếu cần) 

# ### Kiểm tra số nhóm trong mỗi cột
# 
# Như phân tích trên trong dữ liệu thô, chúng ta có 4 cột dưới đây 
# 
# 1. 'Tên sản phẩm'
# 2. 'Level 3'
# 3. 'Level 2'
# 4. 'Level 1'
# 
# Mỗi cột có những tên sản phẩm và nhóm sản phẩm khác nhau. Vì vậy, dưới đây chúng ta sẽ lần lượt tìm hiểu xem trong mỗi cột có bao nhiêu sản phẩm và nhóm sản phẩm
# #### Level 1

# In[16]:


print(Bold + Green+'Số lượng nhóm có trong cột "level 1": {}'.format(len(citigo_data['Level 1'].unique())))
print(Bold + Green+'Danh sách nhóm có trong cột "level 1":'+ End)
print(np.sort(citigo_data['Level 1'].unique()))


# #### Level 2

# In[17]:


print(Bold + Green+'Số lượng nhóm có trong cột "level 2": {}'.format(len(citigo_data['Level 2'].unique())))
print(Bold + Green+'Danh sách nhóm có trong cột "level 2":'+ End)
print(np.sort(citigo_data['Level 2'].unique()))


# #### Level 3

# In[18]:


print(Bold + Green+'Số lượng nhóm có trong cột "level 3": {}'.format(len(citigo_data['Level 3'].unique())))
print(Bold + Green+'Danh sách nhóm có trong cột "level 3":'+ End)
print(np.sort(citigo_data['Level 3'].unique()))


# #### Tên sản phẩm

# In[19]:


print(Bold + Green+'Số lượng mặt hàng có trong cột "Tên sản phẩm": {}'.format(len(citigo_data['Tên sản phẩm'].unique())))
print(Bold + Green+'Danh sách nhóm có trong cột "Tên sản phẩm":'+ End)
print(np.sort(citigo_data['Tên sản phẩm'].unique()))


# $$\textbf{Nhận xét}$$
# 
# Trong bộ dữ liêu của chúng ta thống kê 12,966 sản phẩm. Tất cả các sản phẩm này đều thuộc nhóm 'Sức khỏe - Làm đẹp'. Tiếp đó, chúng lại được phân thành 6 nhóm khác nhau như.
# 
# 1. 'Chăm sóc cơ thể' 
# 2. 'Chăm sóc da' 
# 3. 'Chăm sóc tóc và da đầu' 
# 4. 'Nước hoa'
# 5. 'Tinh dầu spa'
# 6. 'Trang điểm'
# 
# Tiếp đó, chúng lại được xếp loại vào 39 nhóm khác nhau, ví dụ như
# 
# 1. 'Bộ trang điểm' 
# 2. 'Chăm sóc móng' 
# 3. 'Dưỡng Thể, tay chân' 
# 4. 'Dưỡng tóc, ủ tóc'
# 5. 'Dầu gội, dầu xả' 
# 6. 'Dụng cụ tẩy trang' 
# 7. 'Dụng cụ, sản phẩm tạo kiểu tóc'
# 8. etc
# 
# ### Phân tích theo nhóm
# Vì Level 1 chỉ gồm một nhóm hàng là 'Sức khoẻ- Làm đẹp' nên trong trường hợp này để giảm kích thước dữ liệu thuận tiện cho phân tích chúng ta có thê bỏ côt này đi

# In[20]:


citigo_dat=citigo_data.drop('Level 1', axis=1)
citigo_dat.head(5)


# #### Level 2

# In[21]:


level2_citigo_data=citigo_dat.groupby('Level 2')


# In[22]:


print(Bold + Blue + 'Sản phẩn đầu tiên trong 6 nhóm "Level 2":'+ End)
print(level2_citigo_data.first())  


# $\color{Red}{\textbf{Chúng ta thử in ra tất cả các mặt hàng có trong nhóm "Chăm sóc cơ thể"}}$

# In[23]:


print(level2_citigo_data.get_group('Chăm sóc cơ thể'))


# $$\textbf{Nhận xét:}$$
# Chúng ta dễ dàng nhận thấy rằng trong nhóm "Chăm sóc cơ thể" này có tới 1,927 mặt hàng khác nhau. Tương tự, chúng ta có thể áp dụng hàm get_group cho 5 nhóm hàng còn lại. Trong trường hợp này "Level 2" của chúng ta chỉ có 6 nhóm, nên quy trình này sẽ không tốn nhiều thời gian. Tuy nhiên, khi số nhóm lên tới hàng chục, ví dụ như trường hợp của "Level 3" có tới 39 nhóm, thì quy trinh này trở nên tốn thời gian và công sức. Do đó, cách đơn giản nhất là thực hiện một trong hai đoạn mã sau

# In[24]:


print(level2_citigo_data.size())


# In[25]:


print(level2_citigo_data.agg(['count']))


# $$\textbf{Nhận xét:}$$ Cách khác chúng ta sử dụng hàm LOOP như dưới đây

# In[26]:


record_level2 = pd.DataFrame(columns = ['Nhóm_hàng_level_2', 'Số_lượng_tuyệt_đối',
                                        'Số_lượng_tương_đối'])

for i, colum in enumerate(citigo_data['Level 2'].unique()):
    Nhóm_hàng_level_2=colum
    Số_lượng_tuyệt_đối=len(level2_citigo_data.get_group(colum))
    Số_lượng_tương_đối=round(100*len(level2_citigo_data.get_group(colum))/len(citigo_data['Tên sản phẩm'].unique()),
                             2)
                             
    temp_df= pd.DataFrame.from_dict({'Nhóm_hàng_level_2': [Nhóm_hàng_level_2],
                                   'Số_lượng_tuyệt_đối': [Số_lượng_tuyệt_đối],
                                   'Số_lượng_tương_đối': [Số_lượng_tương_đối]})
    #print(Nhóm_hàng_level_2,Số_lượng_tuyệt_đối,Số_lượng_tương_đối)
    record_level2=record_level2.append(temp_df,ignore_index=True)
record_level2.sort_values('Số_lượng_tuyệt_đối',ascending=False)


# $$\textbf{Nhận xét}$$
# Căn cứ Bảng trên chúng ta dễ dàng thấy được, với phân loại hàng theo "Level 2" thành 6 nhóm hàng, thì nhóm "Chăm sóc da" có số lượng mặt hằng nhiều nhất với hơn 6,000 mặt hàng, chiếm tới gần 47 % trong tổng số 12,966 mặt hàng có trong bộ dữ liệu. Tiếp theo là nhóm mặt hàng 'Trang điểm' có tới hơn 2,500 mặt hàng, chiếm hơn 20 %. Ngược lại, nhóm hàng 'Tinh dầu spa' lại có ít số lượng mặt hàng nhất, chỉ có hơn 200 mặt hàng, chiếm chưa đấy 2 % trong tổng số 12,966 mặt hàng có trong mẫu.
# 
# #### Level 3
# 

# In[27]:


level3_citigo_data=citigo_dat.groupby('Level 3')
print(Bold + Blue + 'Sản phẩn đầu tiên trong 39 nhóm hàng theo phân loại "Level 3":'+ End)
level3_citigo_data.first()


#   Bây giờ chúng ta sẽ thống kê số mặt hàng có trong 39 nhóm hàng phân loại theo "Level 3" bằng 1 trong 3 câu lệnh dưới đây

# In[28]:


print(level3_citigo_data.size())


# In[29]:


print(level3_citigo_data.agg(['count']))


# In[30]:


record_level3 = pd.DataFrame(columns = ['Nhóm_hàng_level_3', 'Số_lượng_tuyệt_đối',
                                        'Số_lượng_tương_đối'])

for i, colum in enumerate(citigo_data['Level 3'].unique()):
    Nhóm_hàng_level_3=colum
    Số_lượng_tuyệt_đối=len(level3_citigo_data.get_group(colum))
    Số_lượng_tương_đối=round(100*len(level3_citigo_data.get_group(colum))/len(citigo_data['Tên sản phẩm'].unique()),
                             2)
                             
    temp_df= pd.DataFrame.from_dict({'Nhóm_hàng_level_3': [Nhóm_hàng_level_3],
                                   'Số_lượng_tuyệt_đối': [Số_lượng_tuyệt_đối],
                                   'Số_lượng_tương_đối': [Số_lượng_tương_đối]})
    record_level3=record_level3.append(temp_df,ignore_index=True)
record_level3.sort_values('Số_lượng_tuyệt_đối',ascending=False)


# $$\textbf{Nhận xét:}$$
# 
# Căn cứ Bảng trên, Chúng ta có thể dễ dàng nhìn thấy, theo phân loại nhóm hàng "Level 3", thì nhóm nào có nhiều Mặt Hàng nhất và ít Mặt Hàng Nhất. Cụ thể, nhóm hàng 'Tinh chất dưỡng ẩm & trắng, chống lão hóa' có số lượng Mặt Hàng nhiều nhất, lên tới trên 2,500 Mặt Hàng, chiếm tới 19,39 % trong tổng số 12,966 Mặt Hàng có trong mẫu.
# 
# #### Kết hợp Level 2 và 3
# 
# Chúng ta đã biết, nhóm hàng phân theo Level 2 có 6 nhóm, và nhóm hàng phân theo Level 3 có 39 nhóm. Vậy thì, hai nhóm phân loại này sẽ liên quan như nào với nhau? Trong phần này chúng ta sẽ khai phá xem 39 nhóm hàng trong phân nhóm Level 3 được phân phối như nào vào 6 nhóm hàng trong Level 2

# In[31]:


level23_citigo_data=citigo_dat.groupby(['Level 2','Level 3'])
print(Bold + Blue + 'Sản phẩn đầu tiên trong từng nhóm hàng:'+ End)
level23_citigo_data.first()


# In[32]:


print(Bold + Blue +'Các Nhóm Mặt hàng trong 6 Nhóm Mặt hàng (Level 2):'+ End)
print(level23_citigo_data['Level 2'].nunique())


# $$\textbf{Nhận xét:}$$
# 
# Căn cứ Bảng trên, chúng ta dễ dàng nhìn thấy 39 nhóm hàng theo phân loại Level 3 phân bổ như thế nào vào trong 6 nhóm hàng theo phân loại Level 2. Ví dụ, nhóm hàng 'Tinh dầu spa' thuộc phân loại Level 2 sẽ được phân loại tiếp thành 3 nhóm hàng trong Level 3 như sau
# 
# 1. Sản phẩm tinh dầu.
# 2. Tinh dầu spa khác.
# 3. Đèn/máy xông tinh dầu spa và phụ kiện
# 
# Mặt khác, để thấy được các Mặt hàng có trong từng Nhóm hàng chúng ta chạy Câu Lệnh như ở dưới đây
# 

# In[33]:


print(Bold + Blue +'Phân loại 12,966 Mặt hàng theo nhóm hàng:'+ End)
print(level23_citigo_data.size())


# $$\textbf{Nhận xét:}$$
# 
# Nhìn chung khai phá, phân tích dữ liệu thô bằng phương pháp mô tả thống kê như trên giúp chúng ta hiểu được thêm về sự phân bố của 12,966 Mặt hàng. Tuy nhiên, để rõ ràng hơn về mặt Trực quan và Người sử dụng dễ dàng, nhanh chóng hiểu được thông tin từ Dữ liệu thô thì Phương pháp mô tả bằng Đồ có những ưu điểm vượt trội hơn hẳn. Do đó, phần tiếp theo chúng ta sẽ tiến hành khai khá, phân tích dữ liệu thô bằng Đồ thị hoá.
# 
# ## Phương pháp mô tả bằng đồ thị
# 
# Theo trực quan, thì khai phá dữ liệu thô bằng đồ thị là dễ dàng hiểu nhanh được dữ liệu nhất. Như phân tích ở trên, chúng ta có 12,966 sản phẩm các loại, đều thuộc nhóm 'Sức khỏe - Làm đẹp', trong nhóm này lại được phân thành 6 nhóm mặt hàng nhỏ bên trong và chi tiết hơn nữa là phân loại thành 39 nhóm hàng khác nhau. Do đó, chúng ta hay phân tích sự nhóm hàng theo "Level 2" và "level 3" để biết được sự phổ biến của các nhóm hàng.
# 
# ### Đồ thị dạng cột (bar)
# 
# #### Level 2
# 
# Cách nhanh nhất để xem phân phối của 12,966 Mặt Hàng vào 6 nhóm hàng theo phân loại "Level 2", là chúng ta sử dụng hàm countplot, như dưới đây
# 
# ##### Count plot

# In[34]:


plt.figure(figsize=(12, 5))
sns.countplot(y='Level 2',data=citigo_data,
             orient='h')
plt.grid(which='major',linestyle=':',linewidth=0.9)
plt.title('Phân bổ của 12,966 mặt hàng theo nhóm hàng Level 2',
          fontsize=15,fontweight='bold')


# $$\textbf{Nhận xét}$$
# 
# Cách thứ hai, chúng ta có thể sử dụng kết quả tính toán ở trên và tự vẽ đồ thị theo đoạn mã dưới đây
# 
# ##### Số tuyệt đối

# In[35]:


record_level22=record_level2.set_index(['Nhóm_hàng_level_2'])
fig, ax=plt.subplots(figsize=(10,5)) 
plt.barh(record_level22['Số_lượng_tuyệt_đối'].sort_values(ascending=True).index,
         record_level22['Số_lượng_tuyệt_đối'].sort_values(ascending=True))
plt.autoscale(enable=True, axis='both',tight=True)
plt.title('Phân bổ của 12,966 mặt hàng theo nhóm hàng Level 2',
        fontsize=14, fontweight='bold')
plt.ylabel('Tên Nhóm hàng theo phân loại Level 2',fontsize=10)
plt.xlabel('Số lượng Mặt Hàng', fontsize=10)
plt.grid(which='major',linestyle=':',linewidth=0.9)
for i,v in enumerate(record_level22['Số_lượng_tuyệt_đối'].sort_values(ascending=True)):
    ax.text(v , i-0.15 , str(v), color='blue')#, fontweight='bold')


# $$\textbf{Nhận xét}$$
# 
# 
# Từ đồ thị trên ta thấy, nhóm hàng "Chăm sóc da" chiếm phần lớn, có tới trên 6000 sản phẩm. Đứng thứ hai là nhóm hàng "Trang điểm", với trên 2,500 sản phẩm. Tiếp theo nhóm hàng "Chăm sóc cơ thể" và "Chăm sóc tóc và da đầu", với trên 1,500 mặt hàng. Trong khi đó Cả hai nhóm "Nước hoa" và "Tinh dầu spa" đều có chư tới 500 mặt hàng. Đặc biệt, nhóm "Tinh dầu spa" là ít mặt hàng nhất (217 Mặt Hàng).
# 
# ##### Số tương đối

# In[36]:


fig, ax=plt.subplots(figsize=(10,5)) 
plt.barh(record_level22['Số_lượng_tương_đối'].sort_values(ascending=True).index,
         record_level22['Số_lượng_tương_đối'].sort_values(ascending=True))
plt.autoscale(enable=True, axis='both',tight=True)
plt.title('Phân bổ của 12,966 mặt hàng theo nhóm hàng Level 2',
        fontsize=14, fontweight='bold')
plt.ylabel('Tên Nhóm hàng theo phân loại Level 2',fontsize=10)
plt.xlabel('Tỷ lệ phần trăm (%)', fontsize=10)
plt.grid(which='major',linestyle=':',linewidth=0.9)
for i,v in enumerate(record_level22['Số_lượng_tương_đối'].sort_values(ascending=True)):
    ax.text(v , i-0.15 , str(v), color='blue')#, fontweight='bold')


# $$\textbf{Nhận xét}$$
# 
# 
# Từ đồ thị trên ta thấy, nhóm hàng "Chăm sóc da" chiếm phần lớn, chiếm tới gần 45% trong tổng số 12,966 Mặt Hàng. Đứng thứ hai là nhóm hàng "Trang điểm", chiếm tới gần 21%. Tiếp theo nhóm hàng "Chăm sóc cơ thể" và "Chăm sóc tóc và da đầu", chiếm tới chưa đầy 15%. Trong khi đó Cả hai nhóm "Nước hoa" và "Tinh dầu spa" chiếm tới chưa đầy 5%. Đặc biệt, nhóm "Tinh dầu spa" là ít mặt hàng nhất, chỉ chiếm chưa đầy 2% trong tổng số gần 13,000 Mặt Hàng.
# 
# 
# 
# #### Level 3
# 
# Tương tự như vậy, cách dễ nhanh nhất và đơn giản nhất để xem xem 12,966 Mặt Hàng phân bổ như thế nào vào 39 nhóm Hàng theo tiêu chí "Level 3", chúng ta có thể sử dụng hàm countplot
# 
# ##### Count Plot

# In[37]:


plt.figure(figsize=(12, 10))
sns.countplot(y='Level 3',data=citigo_data,
             orient='h')
plt.title('Phân bổ của 12,966 mặt hàng theo nhóm hàng Level 3',
          fontsize=15,fontweight='bold')
plt.grid(which='major',linestyle=':',linewidth=0.9)


# $$\textbf{Nhận xét:}$$
# 
# Tương tự, chúng ta có thể dựa vào số liệu tính toán ở trên để tự vẽ sự phân bổ của 12,966 Mặt Hàng vào trong 39 nhóm hàng theo phân loại "Level 2" như sau
# 
# ##### Số tuyệt đối

# In[38]:


record_level32=record_level3.set_index(['Nhóm_hàng_level_3'])
fig, ax=plt.subplots(figsize=(12,12)) 
plt.barh(record_level32['Số_lượng_tuyệt_đối'].sort_values(ascending=True).index,
         record_level32['Số_lượng_tuyệt_đối'].sort_values(ascending=True))
plt.autoscale(enable=True, axis='both',tight=True)
plt.title('Phân bổ của 12,966 mặt hàng theo nhóm hàng Level 3',
        fontsize=15, fontweight='bold')
plt.ylabel('Tên Nhóm hàng theo phân loại Level 3',fontsize=12)
plt.xlabel('Số lượng Mặt Hàng', fontsize=12)
plt.grid(which='major',linestyle=':',linewidth=0.9)
for i,v in enumerate(record_level32['Số_lượng_tuyệt_đối'].sort_values(ascending=True)):
    ax.text(v , i-0.15 , str(v), color='blue')


# ##### Số tương đối

# In[39]:


fig, ax=plt.subplots(figsize=(12,12)) 
plt.barh(record_level32['Số_lượng_tương_đối'].sort_values(ascending=True).index,
         record_level32['Số_lượng_tương_đối'].sort_values(ascending=True))
plt.autoscale(enable=True, axis='both',tight=True)
plt.title('Phân bổ của 12,966 mặt hàng theo nhóm hàng Level 3',
        fontsize=15, fontweight='bold')
plt.ylabel('Tên Nhóm hàng theo phân loại Level 3',fontsize=12)
plt.xlabel('Tỷ lệ phần trăm (%)', fontsize=12)
plt.grid(which='major',linestyle=':',linewidth=0.9)
for i,v in enumerate(record_level32['Số_lượng_tương_đối'].sort_values(ascending=True)):
    ax.text(v , i-0.15 , str(v), color='blue')


# $$\textbf{Nhận xét:}$$
# 
# Chúng ta dễ dàng nhận thấy nhóm hàng thuộc về "Tinh chất dưỡng ẩm & trắng, chống lão hoá" có số mặt hàng vượt trội và nhiều nhất so với 38 nhóm hàng còn lại, với trên 2500 mặt hàng. Trong khi các nhóm hàng thuộc về
# 1. "trang điểm mặt", 
# 2. "trang điểm môi", 
# 3. "nước hoa nữ"
# 4. "trang điểm mắt"
# 5. "tinh dầu khác"
# 6. "nước hoa khác" 
# 
# có số sản phẩm ít nhất và gần như là không có sản phẩm nào trong 6 nhóm hàng này.
# 
# ### Đồ thị miếng (Pie)
# 
# #### Level 2

# In[40]:


labels=record_level22.index
fig = plt.figure()
#fig1, ax1 = plt.subplots(figsize=(6, 7))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(record_level22['Số_lượng_tuyệt_đối'],
         labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
#ax1.axis('equal')  
plt.title('Thị phần 6 nhóm sản phẩm phân theo Level 2',fontsize=14, fontweight='bold')
plt.show()


# $$\textbf{Nhận xét:}$$
# 
# Chúng ta có thể tạo ghi chú bên ngoài Đồ thị như đoạn Mã dưới đây, cách vẽ này đặc biệt hữu dụng khi số lượng nhóm lớn như Level 3

# In[41]:


sizes = record_level22['Số_lượng_tuyệt_đối']
labels=record_level22.index
NUM_COLORS = len(record_level22['Số_lượng_tuyệt_đối'])

fig1, ax1 = plt.subplots(figsize=(6, 5))
fig1.subplots_adjust(0.1,0,1,1)

theme = plt.get_cmap('Dark2') #  'prism'
ax1.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes))])

_, _ = ax1.pie(sizes, startangle=90)

ax1.axis('equal')

total = sum(sizes)
plt.legend(
    loc='upper left',
    labels=['%s, %1.1f%%' % (
        l, (float(s) / total) * 100) for l, s in zip(labels, sizes)],
    prop={'size': 11},
    bbox_to_anchor=(0.0, 1),
    bbox_transform=fig1.transFigure
)
plt.title('Thị phần 6 nhóm sản phẩm Level 2',fontsize=14, fontweight='bold')
plt.show()


# $$\textbf{Nhận xét:}$$
# Có vẻ như các cách xây dựng Đồ thị miếng ở trên, ở khía cạnh nào đó, chưa rõ ràng về mặt trực quan, chúng ta hãy thử Câu lệnh dưới đây để xây dựng dạng Đồ thị miếng rõ ràng về trực quan

# In[42]:


fig = px.pie(record_level22, values='Số_lượng_tuyệt_đối', 
             names=record_level22.index, title='Thị phần 6 nhóm sản phẩm Level 2')
fig.show()


# Lưu Đồ thị dưới định dạng .html

# In[130]:


#py.plot(fig, filename='Pie_level_2.html')


# #### Level 3

# In[43]:


labels3=record_level32['Số_lượng_tuyệt_đối'].index
fig1, ax1 = plt.subplots(figsize=(6, 10))
#fig1, ax1 = plt.subplots()
ax1.pie(record_level32['Số_lượng_tuyệt_đối'],
         labels=labels3, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[44]:


sizes = record_level32['Số_lượng_tuyệt_đối']
labels=record_level32.index
NUM_COLORS = len(record_level22['Số_lượng_tuyệt_đối'])

fig1, ax1 = plt.subplots(figsize=(6, 10))
fig1.subplots_adjust(0.1,0,1,1)

theme = plt.get_cmap('prism')
ax1.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes))])

_, _ = ax1.pie(sizes, startangle=90)

ax1.axis('equal')

total = sum(sizes)
plt.legend(
    loc='upper left',
    labels=['%s, %1.1f%%' % (
        l, (float(s) / total) * 100) for l, s in zip(labels, sizes)],
    prop={'size': 11},
    bbox_to_anchor=(0.0, 1),
    bbox_transform=fig1.transFigure
)

plt.show()


# In[45]:


fig = px.pie(record_level32, values='Số_lượng_tuyệt_đối', 
              names=record_level32.index,
             title='Thị phần 39 nhóm sản phẩm Level 3')
#fig.update_traces(textposition='inside', textinfo=record_level32.index)
fig.show()


# In[196]:


#py.plot(fig, filename='Pie_Tuyêt_đôi_Level_3.html')


# $$\textbf{Nhận Xét:}$$
# 
# Chúng ta thấy về mặt trực quan phần ghi chú đã che mất phần số liệu, dó đó, chúng ta dùng Đoạn mã dưới đây để giải quyết việc đó. Kết quả, người dùng chỉ cần di chuột tới phần muốn xem, số liệu sẽ hiện ra

# In[200]:


fig = px.pie(record_level32, values='Số_lượng_tương_đối', 
              names=record_level32.index,
             title='Thị phần 39 nhóm sản phẩm Level 3')
fig.show()


# In[46]:


#py.plot(fig, filename='Pie_Tương_Đối_Level_3.html')


# ### Đồ thị dang Donut
# #### Level 2

# In[47]:


fig = go.Figure(data=[go.Pie(labels=record_level22.index, 
                             values=record_level22['Số_lượng_tuyệt_đối'],
                             hole=.3)])
#fig.update(layout_title_text='Thị phần 6 nhóm sản phẩm phân theo Level 2',
 #          layout_showlegend=True)

fig.update_layout(
    title_text='Thị phần 6 nhóm sản phẩm phân theo Level 2',
    annotations=[dict(text='Level 2', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()


# In[138]:


#py.plot(fig, filename='Donut_Level_2.html')


# #### Level 3

# In[48]:


fig = go.Figure(data=[go.Pie(labels=record_level32.index, 
                             values=record_level32['Số_lượng_tương_đối'],
                             hole=.3)])
fig.update_layout(title_text='Thị phần 39 nhóm sản phẩm (Level 3)',
    annotations=[dict(text='Level 3', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()


# Chúng ta có thể lưu đồ thị trên ở định dạng .html như dưới đây

# In[140]:


#py.plot(fig, filename='Donut_Level_3.html')


# #### Kết hợp 3 Levels
# Như vậy, thông qua các dạng Đồ thị dạng Miếng (Pie chart) và dạng bánh Donut (Donut chart), chúng ta đã phân tích được sự phân bổ của 12,966 Mặt hàng vào các Nhóm hàng phân loại kiểu Level 2 và Level 3 một cách riêng rẽ. Để có được cái nhìn tổng quan nhất về sự phân phối của 12,966 Mặt hàng vào trong các Nhóm hàng khác nhau, bao gồm tất cả các mức độ phân loại nhóm từ Level 1, Level 2, và Level 3, chúng ta sẽ tiến hành xây dựng Đồ thị dạng hình bánh Donut, theo các tuần tự các bước như sau.
# 
# ##### Nhóm toàn bộ Mặt hàng
# 
# 

# In[49]:


#citigo_data.head(5)
level123_citigo_data=citigo_data.groupby(['Level 1','Level 2','Level 3'])
print(Bold + Blue + 'Sản phẩn đầu tiên trong từng nhóm hàng:'+ End)
level123_citigo_data.first()


# Tiếp theo chúng ta sẽ tiến hành, thống kê các Mặt hàng vào trong các Nhóm hàng. Sau đó, dữ liệu sẽ được lưu vào một Mảng dữ liệu riêng 
# ##### Thống kê các Mặt hàng
# 
# 

# In[50]:


level123=level123_citigo_data.size()  #agg(['count'])
print(level123)


# Tiếp theo, Chúng ta tiến hành chuyển đội dạng dữ liệu đươc lưu, sang định dạng dữ liệu của Pandas trong một Vùng dữ liệu mới
# 
# ##### Chuyển đổi định dạng lưu dữ liệu

# In[51]:


level23_pan=level123.reset_index() 
level23_pan=level23_pan.rename(columns={0: "Số mặt hàng"})
print(Bold + Blue + 'Tên các cột trong bảng dữ liệu:'+ End)
print(level23_pan.columns)
print(Bold + Blue + 'Dữ liệu của 5 quan sát đầu tiên:'+ End)
level23_pan.head(5)


# Cuối cùng, chúng ta dựng đồ thị như sau
# ##### Dựng đồ thị

# In[52]:


fig = px.sunburst(level23_pan, path=['Level 1', 'Level 2', 'Level 3'], values='Số mặt hàng')
fig.update_layout(title_text='Thị phần tuyệt đối của 12,966 Mặt hàng phân theo các Cấp Nhóm hàng')
fig.show()


# In[145]:


#py.plot(fig, filename='Donut_Tuyệt_Đối_Level_2_3.html')


# $$\textbf{Hướng dẫn đọc dữ liệu:}$$
# 
# Để xem có bao nhiêu Mặt hàng được phân bổ trong các Nhóm hàng theo các cấp độ, Người sử dụng chỉ cần thực hiện bước đơn giản là di chuyển Con Chuột Máy tính với Nhóm Mặt hàng trên Đồ thị. Thông tin sẽ được hiện ra. Tuy nhiên, Đồ thị dạng hình bánh Donut trên chỉ cho biết số lượng tuyệt đối các Mặt hàng và không hiển thị thị phần tương đối của chúng. Để biết được thị phần tương đối của gần 13,000 Mặt hàng, chúng ta thực hiện các Câu lệnh tiếp theo như dưới đây
# 
# ###### Tính thị phần các mặt hàng theo tỷ lệ phần trăm

# In[53]:


relative=level23_pan['Số mặt hàng']/level23_pan['Số mặt hàng'].sum(axis = 0, skipna = True) 

relative=round(100*relative,2)
#relative

level23_pan['Thị phần (%)']=relative
level23_pan.head(5)


# ##### Dựng đồ thị

# In[54]:


fig = px.sunburst(level23_pan, path=['Level 1', 'Level 2', 'Level 3'], values='Thị phần (%)')
fig.update_layout(title_text='Thị phần tương đối (%) của 12,966 Mặt hàng phân theo các Cấp Nhóm hàng')
fig.show()


# Chúng ta có thể lưu đồ thị trên ở định dạng .html như dưới đây

# In[148]:


#py.plot(fig, filename='Donut_Tương_Đối_Level_2_3.html')


# # Kết luận
# 
# Như vậy, bằng cả phương pháp thống kê mô tả và đồ thị, chúng ta đã thấy được Bức tranh toàn cảnh về thị phần của gần 13,000 Mặt hàng về Sức khoẻ-Làm đẹp trên thị trường.
# 
# Xin cảm ơn các bạn đã dành thời gian đọc và xem toàn bộ quy trình thực hiện. Mọi chi tiết câu hỏi, nhận xét và đóng góp xin các bạn vui lòng gửi về theo địa chỉ thư điện tử dưới đây.
# 
# 
# 
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# $\Large \color{Blue}{\textbf{ --------------------------------------------------------------------------------------}}$
# 
# $$\small \color{red}{\textbf{The End}}$$
# 
# 
# $\Large \color{Blue}{\textbf{ --------------------------------------------------------------------------------------}}$
