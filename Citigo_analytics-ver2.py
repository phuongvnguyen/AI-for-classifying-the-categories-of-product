#!/usr/bin/env python
# coding: utf-8

# $$\Large \color{green}{\textbf{Phát Triển Chương Trình Máy Học/AI Cho Phân loại hàng hoá}}$$
# $$\Large \color{green}{\textbf{Dự Án của Citigo, Kiotviet}}$$
# 
# 
# $$\large \color{blue}{\textbf{Phuong Van Nguyen}}$$
# $$\small \color{red}{\textbf{ phuong.nguyen@summer.barcelonagse.eu}}$$
# 
# $\color{green}{\underline{\textbf{I. Giới thiệu Chung}}}$
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
# 
# $\color{green}{\underline{\textbf{II. Giới thiệu quy trình chung cho đào tạo một chương trình Máy Học}}}$
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
# $\color{green}{\underline{\textbf{III. Giới thiệu vấn đề}}}$
# 
# Kiotviet có một bộ dữ liệu lên tới Mấy trăm ngàn Mặt Hàng. Mỗi Mặt hàng đều được ghi Tên cụ thể. Vì vậy vấn đề đặt ra như sau:
# 
# Với một số lượng lớn dữ liệu Tên Mặt hàng như vậy, làm thế nào có thể phân loại chúng thành các tiêu chí khác nhau.
# 
# Để làm điều này, chúng ta cần phát triển một Chương trình Máy Học hoặc AI. Theo đó, chúng có khả năng chỉ cần dựa vào Tên Sản phẩm, chúng có thể phân loại sản phẩm đó vào các Nhóm hàng theo tiêu chí Level 1, Level 2, và Level 3.
# 
# $\color{green}{\underline{\textbf{IV. Giới thiệu dữ liệu}}}$
# 
#  Bộ dữ liệu liệt kê tên của hơn 13,000 Mặt Hàng, và các tiêu chí phân loại Nhóm của hơn 13,000 Mặt hàng đó. Toàn bộ dữ liệu thô, có thể tải theo đường dẫn dưới đây
#  
#  https://github.com/phuongvnguyen/Kiotviet-Data-Analytics/blob/master/Product%20Classify.xlsx
# 
# $\color{green}{\underline{\textbf{V. Kết quả phân tích dữ liệu thô}}}$
# 
# Kết quả phân tích dữ liệu dựa trên cả hai phương pháp là Phân tích thông kê mô tả và Đồ thị (dạng cột, dạng Pie, và dạng Donut).  Đặc biệt, Phương pháp phân tích bằng Đồ thị dạng Donut,
# 
# 
# ![Screenshot%202020-03-06%2008.00.26.png](attachment:Screenshot%202020-03-06%2008.00.26.png)
# 
# https://github.com/phuongvnguyen/Kiotviet-Data-Analytics/blob/master/Donut_Tuye%CC%A3%CC%82t_%C4%90o%CC%82%CC%81i_Level_2_3.html
# 
# 
# ![Screenshot%202020-03-06%2008.02.24.png](attachment:Screenshot%202020-03-06%2008.02.24.png)
# 
# 
# 
# https://github.com/phuongvnguyen/Kiotviet-Data-Analytics/blob/master/Donut_Tu%CC%9Bo%CC%9Bng_%C4%90o%CC%82%CC%81i_Level_2_3.html
# 
# cho chúng ta được cách nhìn tổng quan toàn bộ phân phối và thị phần của các nhóm hàng. Chi tiết các bạn có thể xem các Đồ thị này và các đoạn Câu lệnh trong chương trình. Một vài điểm phân tích nổi bật được đưa ra như sau.
# 
# Bộ Dữ liệu thô, thống kê gần 13,000 Mặt hàng đều thuộc về Nhóm hàng $\textbf{Sức khoẻ - làm đẹp}$ (phân theo tiêu chí Level 1). Đồ thị dạng Donut cho chúng ta thấy, trong Nhóm hàng Sức khoẻ - làm Đẹp này, lại được phân loại thành 6 Nhóm hàng nhỏ tiếp theo (phân loại theo tiêu chí Level 2), bao gồm:
# 
# 1. $\textbf{Chăm sóc da}$: có số lượng Mặt hàng nhiều nhất, với hơn 6,000 Mặt hàng, chiếm gần 47% Thị phần toàn Mặt hàng. Đồng thời, hơn 6,000 Mặt hàng Chăm sóc da, tiếp tục được phân loại vào 8 Nhóm Mặt hàng (phân theo Level 3). Ví dụ, nhóm hàng $\textbf{Tinh chất dưỡng ẩm & trắng, chống lão hoá}$, có tới hơn 2,500 Mặt hàng, chiếm hơn 19% thị phần, cao nhất trong toàn Mặt hàng.
# 
# 2. $\textbf{Trang điểm}$: Có số lượng Mặt hàng nhiều thứ hai, với gân 2,700 Mặt hàng, chiếm gần 21% Thị phần toàn Mặt hàng. Thêm nữa, gần 2,700 Mặt hàng Trang điểm này, lại được phân vào 7 Nhóm hàng nhỏ tiếp theo (phân loại theo Level 3), như $\textbf{Trang điểm môi}$ (862 Mặt hàng:6.62%), $\textbf{Trang điểm mắt}$ (533 Mặt hàng:4.09%), $\textbf{Trang điểm mắt}$ (428 Mặt hàng: 3.29%), etc 
# 
# 3. $\textbf{Chăm sóc cơ thể}$: Có số lượng Mặt hàng nhiều thứ 3, hơn 1,900 Mặt hàng, chiếm gần 15% Thị phần toàn Mặt hàng. Đồng thời nhóm hàng Chăm sóc cơ thể này, lại được chia làm 6 nhóm hàng nhỏ (phân loại theo Level 3), như $\textbf{Sữa tắm, xà bông, xà bông, tẩy tế bào chết cơ thể}$ (642 Mặt hàng: 4.93%), $\textbf{Sản phẩm khử mùi}$ (300 Mặt hàng: 2.3%), etc.
# 
# 4. $\textbf{Chăm sóc da tóc và da dầu}$: Có số lượng Mặt hàng nhiều thứ 4, với gần 1,700 Mặt hàng, chiếm gần 13% thị phần toàn Mặt hàng. Đồng thời, gần 1,700 Mặt hàng Chăm sóc da tóc và da dầu, lại tiếp tục được phân loại vào 7 Nhóm mặt hàng (theo tiêu chí phân loại Level 3), chẳng hạn như $\textbf{Dầu gội, dầu xả}$ (920 Mặt hàng: 7.06%), $\textbf{Dưỡng tóc, ủ tóc}$ (190 Mặt hàng: 1.46%), etc.
# 
# 5. $\textbf{Nước hoa}$: Có số lượng Mặt hàng ít thứ hai, với 405 Mặt hàng, chiếm  3.12% thị phần toàn Mặt hàng, bao gồm ba nhóm nước hoa, như $\textbf{Nước hoa nữ}$ (38 Mặt hàng: 1.77%), $\textbf{Nước hoa nam}$ (87 Mặt hàng: 0.67%), và $\textbf{Nước hoa khác}$ (86 Mặt hàng: 0.66%).
# 
# 6. $\textbf{Tinh dầu spa}$: Có số lượng Mặt hàng ít nhất, 217 Mặt hàng, chiếm chưa đầy 2% Thị phần toàn Mặt hàng.
# 
# Mặt, khác Đồ thị dạng cột
# 
# ![output_67_0.png](attachment:output_67_0.png)
# 
# 
# ![output_73_0.png](attachment:output_73_0.png)
# 
# 
# 
# cho chúng ta thấy, theo phân loại nhóm hàng tiêu chí Level 3, thì Nhóm hàng có số mặt hàng nhiều nhất thuộc về $\textbf{Tinh chất dưỡng ẩm & trắng, chống lão hoá}$, có tới hơn 2,500 Mặt hàng, chiếm hơn 19% thị phần. Tiếp theo là Nhóm hàng $\textbf{Dầu gội, dầu xả}$ (920 Mặt hàng: 7.1%), $\textbf{Sản phẩm chăm sóc da khác}$ (896 Mặt hàng: 6.91%), etc. Cuối cùng, nhóm mặt hàng có ít số lượng sản phẩm nhất, thuộc về Nước hoa, tinh dầu spa khác, và trang điểm mắt, có chưa tới 10 Mặt hàng, thị phần chưa đầy 1% toàn Mặt hàng.
# 
# 
# $\color{green}{\underline{\textbf{VI. Quy Trình và Kết quả đào tạo Chương Trình Máy Học}}}$
# 
# Sau khi xử lý xong phần dữ liệu thô, như lựa chọn biến đầu vào là "Tên sản phẩm" được Mã hoá thành các giá trị TF-IDF và biến đầu ra là các Nhóm hàng "Level 2, 3" được mã hoá thành hai giá 0 và 1, etc. 
# 
# ![output_134_1.png](attachment:output_134_1.png)
# 
# Một loạt các Thuật toán Máy học được đào tạo trên bộ dữ liệu này và so sánh hiệu quả hoạt động với nhau. Cuối cùng, Chương trình Máy học với Thuật toán linearSVC được lựa chọn, bởi Nó cho kết quả dự báo chính xác cao nhất (gần 95%). Hiệu quả hoạt động của Chương trình Máy học này sau đó tiếp tục được nâng cao bằng cách sử dụng lần lượt các phương pháp sau:
# 
# 1. Tối ưu hoá Thuật toán dựa trên Dữ liệu (model tunning).
# 2. Sử dụng phương pháp Ensemble.
# 3. Sử dụng phương pháp Voting.
# 4. Sử dụng các Thuật toán AI.
# 
# $\color{green}{\underline{\textbf{VII. Hạn chế và Cách khắc phục}}}$
# 
# $\textbf{Hạn chế của Dữ Liệu}$
# 
# 1. Như phân tích ở trên Bộ Dữ liệu có sự mất cân đối cao, đặc biệt những Nhóm có số lượng Mặt hàng rất ít, chẳng hạn ở Nhóm Level 2 có 6 Nhóm hàng khác nhau, thì Nhóm Nước Hoa và Tinh Dầu Spa, số Mặt hàng chỉ có chưa tới 5% tổng số Mặt hàng. Hoặc ở Nhóm Level 3, có tới 39 Nhóm hàng khác nhau, thì nhiều Nhóm có số sản phẩm chiếm chưa tới 1% Tổng sản phẩm. Những hạn chế này, sẽ khiến cho Chương trình Máy Học/AI rất khó được đào tạo để có thể nhận biết được các sản phẩm thuộc vào nhóm này, khi chỉ dựa vào Tên sản phẩm.
# 
# 2. Tên sản phẩm cần được làm rõ hơn, bởi có nhiều sản phẩm ghi chép không rõ ý nghĩa, như "xã sUuop" hay ghi cả Tên sản phẩm cả bằng Tiếng Hàn, etc. Dó đó, cần làm "sạch" bộ dữ liệu thô là cần thiết.
# 
# $\textbf{Hạn chế của Máy Học/AI}$
# 
# 1. Tính toán lại TF-IDF, bởi như đã nói ở trên, còn tồn tại những Chữ không có ý nghĩa cho đặt "Tên sản phẩm" khi tính lại TF-IDF cần xem xét sự tồn tại những Chữ này trong việc đặt Tên sản phẩm. Giải pháp, đặt cấu hình cho Tham số "Stop words" trong Hàm tính TF-IDF, và sử dụng thư viện Ngôn ngữ NTDK (https://www.nltk.org/index.html), là một trong những cách để tìm hiểu "Stop words" cho Tiếng Việt.
# 
# 2. Chương trình Máy học mới thử nghiệm trên một Nhóm sản phẩm là ['Level 2_Chăm sóc cơ thể'] Do đó, chương trình Máy Học nên được đào tạo trên các Nhóm còn lại. Để làm điều này, tạo hàm LOOP (for in).
# 
# 
# 3. Chương trình mới đào tạo và so sánh 5 Thuật toán khác nhau, do đó, cần bổ sung đào tạo thêm các Thuật toán Máy học khác.
# 
# 4. Các Thuật Toán Ensemble hiện đại chưa được áp dụng và đào tạo trên bộ dữ liệu, như Xgboost, hay lightGBM, nên chưa thực sự thấy vai trò của chúng trong việc nâng cao hiệu quả Chương trình Máy học. Hơn nữa, mặc dù Thuật toán Ramdon Forest Classifier (RFC) được lựa chọn, nhưng nó chưa được tối ưu hoá hoàn toàn, với lý do, Máy tính các nhân không thể làm điều này. Do đó, giải pháp là nên sử dụng Máy tính có Cấu hình cao nhất hoặc sử dụng dịch vụ Máy Chủ Amazon Web Server (AWS) (không hề rẻ cho Cá Nhân LOL, nếu bất cẩn quên tắt kết nối thì chịu một số tiền CỰC SHOCK).
#  
# 5. Chương trình Máy học chưa chỉ ra những Chữ mà nó khó nhận biết để phân loại chúng vào các Nhóm hàng Level 2 và 3. Giải pháp là tạo Đám Mây Chữ (Words Cloud). Điều này tương tự như, tạo ra Đám Mây chữ mà có số Chữ có liên quan nhiều nhất tới Nhóm sản phẩm cần phân loại. Ví dụ, hình dưới đây, là một Đám Mây chữ cho biết những Chữ có liên quan nhiều nhất tới Nhóm sản phẩm Level 3 Bộ Trang Điểm và Level 2 Trang điểm.
# 
# ![output_148_0.png](attachment:output_148_0.png)
# 
# ![output_150_5.png](attachment:output_150_5.png)
# 
# 5. AI được đào tạo trên bộ dữ liệu Thô với nhiều sai sót chính tả của toàn bộ 45 Nhóm Hàng, và đưa dự báo chính xác được trên 95%, tuy nhiên, bản thân nó vẫn còn tồn tại những Hạn chế, sẽ được nêu ra trong phần Kết Luận.
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
import itertools
#from time import time
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from scipy import stats
from pandas import set_option
from pandas.plotting import scatter_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


# ## Các thuật toán Máy Học
# ### Thuật toán tuyến tính

# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import PassiveAggressiveClassifier


# ### Thuật toán phi tuyến tính

# In[3]:


from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
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
#import lightgbm as lgb
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


# ## Deep Learning

# In[312]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# ## Phương pháp đánh giá mô hình

# In[8]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# ##  Định nghia biến cho in kết quá

# In[9]:


from pickle import dump
from pickle import load
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
from pprint import pprint


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

# In[329]:


citigo_data.head(10)


# ### Kiểm tra kích cỡ dữ liệu

# In[330]:


print('--------------------')
print(Bold+ Red  + 'Kích cỡ dữ liệu thô:' + End)
print('--------------------')
print(Bold + Blue + 'Số quan sát: {}'.format(len(citigo_data)))
print(Bold + Blue + 'Số cột: {}'.format(len(citigo_data.columns)))
print(Bold+ Blue + 'Danh sách cột:' + End)
print(citigo_data.columns)
print('--------------------')


# ### Kiểm tra thiếu dữ liệu

# In[331]:


print(Bold + Darkcyan +'Số dữ liêu bị thiếu:'+ End)
print(citigo_data.isnull().sum())


# $$\textbf{Nhận xét:}$$
# 
# Từ kết quả in ra ở trên cho thấy rất may mắn là dữ liệu thô của chúng ta không có cột nào bị thiếu dữ liệu.

# ### Kiểm tra dạng dữ liệu

# In[332]:


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

# In[333]:


print(Bold + Green+'Số lượng nhóm có trong cột "level 1": {}'.format(len(citigo_data['Level 1'].unique())))
print(Bold + Green+'Danh sách nhóm có trong cột "level 1":'+ End)
print(np.sort(citigo_data['Level 1'].unique()))


# #### Level 2

# In[334]:


print(Bold + Green+'Số lượng nhóm có trong cột "level 2": {}'.format(len(citigo_data['Level 2'].unique())))
print(Bold + Green+'Danh sách nhóm có trong cột "level 2":'+ End)
print(np.sort(citigo_data['Level 2'].unique()))


# #### Level 3

# In[335]:


print(Bold + Green+'Số lượng nhóm có trong cột "level 3": {}'.format(len(citigo_data['Level 3'].unique())))
print(Bold + Green+'Danh sách nhóm có trong cột "level 3":'+ End)
print(np.sort(citigo_data['Level 3'].unique()))


# #### Tên sản phẩm

# In[336]:


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

# In[337]:


citigo_dat=citigo_data.drop('Level 1', axis=1)
citigo_dat.head(5)


# #### Level 2

# In[338]:


level2_citigo_data=citigo_dat.groupby('Level 2')


# In[339]:


print(Bold + Blue + 'Sản phẩn đầu tiên trong 6 nhóm "Level 2":'+ End)
print(level2_citigo_data.first())  


# $\color{Red}{\textbf{Chúng ta thử in ra tất cả các mặt hàng có trong nhóm "Chăm sóc cơ thể"}}$

# In[340]:


print(level2_citigo_data.get_group('Chăm sóc cơ thể'))


# $$\textbf{Nhận xét:}$$
# Chúng ta dễ dàng nhận thấy rằng trong nhóm "Chăm sóc cơ thể" này có tới 1,927 mặt hàng khác nhau. Tương tự, chúng ta có thể áp dụng hàm get_group cho 5 nhóm hàng còn lại. Trong trường hợp này "Level 2" của chúng ta chỉ có 6 nhóm, nên quy trình này sẽ không tốn nhiều thời gian. Tuy nhiên, khi số nhóm lên tới hàng chục, ví dụ như trường hợp của "Level 3" có tới 39 nhóm, thì quy trinh này trở nên tốn thời gian và công sức. Do đó, cách đơn giản nhất là thực hiện một trong hai đoạn mã sau

# In[341]:


print(level2_citigo_data.size())


# In[342]:


print(level2_citigo_data.agg(['count']))


# $$\textbf{Nhận xét:}$$ Cách khác chúng ta sử dụng hàm LOOP như dưới đây

# In[343]:


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

# In[344]:


level3_citigo_data=citigo_dat.groupby('Level 3')
print(Bold + Blue + 'Sản phẩn đầu tiên trong 39 nhóm hàng theo phân loại "Level 3":'+ End)
level3_citigo_data.first()


#   Bây giờ chúng ta sẽ thống kê số mặt hàng có trong 39 nhóm hàng phân loại theo "Level 3" bằng 1 trong 3 câu lệnh dưới đây

# In[345]:


print(level3_citigo_data.size())


# In[346]:


print(level3_citigo_data.agg(['count']))


# In[347]:


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

# In[348]:


level23_citigo_data=citigo_dat.groupby(['Level 2','Level 3'])
print(Bold + Blue + 'Sản phẩn đầu tiên trong từng nhóm hàng:'+ End)
level23_citigo_data.first()


# In[349]:


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

# In[350]:


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

# In[351]:


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

# In[352]:


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

# In[353]:


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

# In[354]:


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

# In[355]:


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

# In[356]:


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

# In[357]:


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

# In[358]:


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

# In[359]:


fig = px.pie(record_level22, values='Số_lượng_tuyệt_đối', 
             names=record_level22.index, title='Thị phần 6 nhóm sản phẩm Level 2')
fig.show()


# Lưu Đồ thị dưới định dạng .html

# In[360]:


#py.plot(fig, filename='Pie_level_2.html')


# #### Level 3

# In[361]:


labels3=record_level32['Số_lượng_tuyệt_đối'].index
fig1, ax1 = plt.subplots(figsize=(6, 10))
#fig1, ax1 = plt.subplots()
ax1.pie(record_level32['Số_lượng_tuyệt_đối'],
         labels=labels3, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[362]:


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


# In[363]:


fig = px.pie(record_level32, values='Số_lượng_tuyệt_đối', 
              names=record_level32.index,
             title='Thị phần 39 nhóm sản phẩm Level 3')
#fig.update_traces(textposition='inside', textinfo=record_level32.index)
fig.show()


# In[364]:


#py.plot(fig, filename='Pie_Tuyêt_đôi_Level_3.html')


# $$\textbf{Nhận Xét:}$$
# 
# Chúng ta thấy về mặt trực quan phần ghi chú đã che mất phần số liệu, dó đó, chúng ta dùng Đoạn mã dưới đây để giải quyết việc đó. Kết quả, người dùng chỉ cần di chuột tới phần muốn xem, số liệu sẽ hiện ra

# In[365]:


fig = px.pie(record_level32, values='Số_lượng_tương_đối', 
              names=record_level32.index,
             title='Thị phần 39 nhóm sản phẩm Level 3')
fig.show()


# In[366]:


#py.plot(fig, filename='Pie_Tương_Đối_Level_3.html')


# ### Đồ thị dang Donut
# #### Level 2

# In[367]:


fig = go.Figure(data=[go.Pie(labels=record_level22.index, 
                             values=record_level22['Số_lượng_tuyệt_đối'],
                             hole=.3)])
#fig.update(layout_title_text='Thị phần 6 nhóm sản phẩm phân theo Level 2',
 #          layout_showlegend=True)

fig.update_layout(
    title_text='Thị phần 6 nhóm sản phẩm phân theo Level 2',
    annotations=[dict(text='Level 2', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()


# In[368]:


#py.plot(fig, filename='Donut_Level_2.html')


# #### Level 3

# In[369]:


fig = go.Figure(data=[go.Pie(labels=record_level32.index, 
                             values=record_level32['Số_lượng_tương_đối'],
                             hole=.3)])
fig.update_layout(title_text='Thị phần 39 nhóm sản phẩm (Level 3)',
    annotations=[dict(text='Level 3', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()


# Chúng ta có thể lưu đồ thị trên ở định dạng .html như dưới đây

# In[370]:


#py.plot(fig, filename='Donut_Level_3.html')


# #### Kết hợp 3 Levels
# Như vậy, thông qua các dạng Đồ thị dạng Miếng (Pie chart) và dạng bánh Donut (Donut chart), chúng ta đã phân tích được sự phân bổ của 12,966 Mặt hàng vào các Nhóm hàng phân loại kiểu Level 2 và Level 3 một cách riêng rẽ. Để có được cái nhìn tổng quan nhất về sự phân phối của 12,966 Mặt hàng vào trong các Nhóm hàng khác nhau, bao gồm tất cả các mức độ phân loại nhóm từ Level 1, Level 2, và Level 3, chúng ta sẽ tiến hành xây dựng Đồ thị dạng hình bánh Donut, theo các tuần tự các bước như sau.
# 
# ##### Nhóm toàn bộ Mặt hàng
# 
# 

# In[371]:


#citigo_data.head(5)
level123_citigo_data=citigo_data.groupby(['Level 1','Level 2','Level 3'])
print(Bold + Blue + 'Sản phẩn đầu tiên trong từng nhóm hàng:'+ End)
level123_citigo_data.first()


# Tiếp theo chúng ta sẽ tiến hành, thống kê các Mặt hàng vào trong các Nhóm hàng. Sau đó, dữ liệu sẽ được lưu vào một Mảng dữ liệu riêng 
# ##### Thống kê các Mặt hàng
# 
# 

# In[372]:


level123=level123_citigo_data.size()  #agg(['count'])
print(level123)


# Tiếp theo, Chúng ta tiến hành chuyển đội dạng dữ liệu đươc lưu, sang định dạng dữ liệu của Pandas trong một Vùng dữ liệu mới
# 
# ##### Chuyển đổi định dạng lưu dữ liệu

# In[373]:


level23_pan=level123.reset_index() 
level23_pan=level23_pan.rename(columns={0: "Số mặt hàng"})
print(Bold + Blue + 'Tên các cột trong bảng dữ liệu:'+ End)
print(level23_pan.columns)
print(Bold + Blue + 'Dữ liệu của 5 quan sát đầu tiên:'+ End)
level23_pan.head(5)


# Cuối cùng, chúng ta dựng đồ thị như sau
# ##### Dựng đồ thị

# In[374]:


fig = px.sunburst(level23_pan, path=['Level 1', 'Level 2', 'Level 3'], values='Số mặt hàng')
fig.update_layout(title_text='Thị phần tuyệt đối của 12,966 Mặt hàng phân theo các Cấp Nhóm hàng')
fig.show()


# In[375]:


#py.plot(fig, filename='Donut_Tuyệt_Đối_Level_2_3.html')


# $$\textbf{Hướng dẫn đọc dữ liệu:}$$
# 
# Để xem có bao nhiêu Mặt hàng được phân bổ trong các Nhóm hàng theo các cấp độ, Người sử dụng chỉ cần thực hiện bước đơn giản là di chuyển Con Chuột Máy tính với Nhóm Mặt hàng trên Đồ thị. Thông tin sẽ được hiện ra. Tuy nhiên, Đồ thị dạng hình bánh Donut trên chỉ cho biết số lượng tuyệt đối các Mặt hàng và không hiển thị thị phần tương đối của chúng. Để biết được thị phần tương đối của gần 13,000 Mặt hàng, chúng ta thực hiện các Câu lệnh tiếp theo như dưới đây
# 
# ###### Tính thị phần các mặt hàng theo tỷ lệ phần trăm

# In[376]:


relative=level23_pan['Số mặt hàng']/level23_pan['Số mặt hàng'].sum(axis = 0, skipna = True) 

relative=round(100*relative,2)
#relative

level23_pan['Thị phần (%)']=relative
level23_pan.head(5)


# ##### Dựng đồ thị

# In[377]:


fig = px.sunburst(level23_pan, path=['Level 1', 'Level 2', 'Level 3'], values='Thị phần (%)')
fig.update_layout(title_text='Thị phần tương đối (%) của 12,966 Mặt hàng phân theo các Cấp Nhóm hàng')
fig.show()


# Chúng ta có thể lưu đồ thị trên ở định dạng .html như dưới đây

# In[378]:


#py.plot(fig, filename='Donut_Tương_Đối_Level_2_3.html')


# 
# 
# 
# # Biến đổi dữ liệu thô
# 
# Bài toán đặt ra là, phát triển một Chương trình Máy học, chỉ dựa vào dựa theo Tên của Sản phẩm để chúng phân tách các sản phẩm đó vào các Nhóm sản phẩm Level 2 (39 Nhóm) và Level 3 (6 Nhóm). Do đó, Tên sản phẩm sẽ là biến giải thích, các gía trị đầu ra sẽ là Nhóm sản phẩm Level 2 và Level 3. Tuy nhiên chúng ta đã biết, dữ liệu thô của chúng ta đều ở dạng Văn bản, nên trước khi đào tạo một chương trình Máy Học, chúng ta cần chuyển đổi chúng thành dạng số, như sau
# 
# 1. Với biến đầu ra/biến phụ thuộc: biến đổi chúng thành dạng Binary.
# 2. Với biến đầu vào là "Tên sản phẩm": biến đổi chúng thành dạng TF-IDF
# 
# ## Chuyển đổi sang dạng số cho biến đầu ra
# 
# Các cột Phân loại Mặt hàng theo tiêu chí "Level 2" và "Level 3" đều ở dạng "Chữ" (Categorical data). Do đó, trước khi tiến hành đào tạo một chương trình Máy học hoặc AI, chúng ta tiến hành chuyển đối dạng dữ liệu phân loại này sang dạng số (numerical data), bằng cách tạo ra biến giả cho chúng như dưới đây

# In[379]:


citigo_dummy=pd.get_dummies(citigo_data, columns=['Level 3', 'Level 2'], drop_first=False)
print(Bold+ Blue+'Kích thước của dữ liệu mới:'+End)
print('Số lượng sản phẩm: %d; Số lượng Cột: %d'%(citigo_dummy.shape))
print(Bold+ Blue+'3 Mặt Hàng đầu tiên:'+End)
print(citigo_dummy.head(3))


# $$\textbf{Nhận xét:}$$
# Chúng ta dễ dàng nhận thấy, dữ liệu sau biến đổi có số lượng sản phẩm như cũ là 13,024 Mặt Hàng, nhưng có tới 47 cột, so với 4 Cột trong dữ liệu gốc. Điều này là bởi vì, chúng ta đã biến đổi 39 Nhóm hàng trong Level 3 và 6 Nhóm hàng trong Level 2, thành các biến dạng số ở dạng Binary. 
# 
# Mặt khác, do trong trường hợp này, chúng ta không quan tâm tới phân loại sản phẩm theo tiêu chí Level 1, nên chúng ta ỏ cột Level 1 đi
# 
# ## Loại Level

# In[380]:


citigo_dummy=citigo_dummy.drop('Level 1', axis=1)
citigo_dummy.head(3)


# $$\textbf{Nhận xét:}$$
# 
# Như vậy, chúng ta có tới 45 biến đầu ra, và biến đầu vào là "Tên sản phẩm"
# 
# # Khai phá dữ liệu mới
# 
# ## Thống kê mô tả
# 
# ### Lấy mẫu tên Sản phẩm

# In[15]:


citigo_dummy['Tên sản phẩm'][0]


# In[16]:


citigo_dummy['Tên sản phẩm'][2]


# ### Thống kê các nhóm SP

# In[17]:


print(Bold + Blue + 'Thống kê mô tả các nhóm sản phẩm' + End)
print(citigo_dummy.describe())


# ### Đo và thống kê độ dài tên sản phẩm
# #### Đo độ dài của tên

# In[18]:


lens =citigo_dummy['Tên sản phẩm'].str.len()
print(Bold+ Blue +'Độ dài tên của 5 sản phẩm đầu tiên:' +End)
print(lens.head(5))


# #### Thống kê biến động độ dài tên SP
# ##### Thống kê mô tả

# In[19]:


print(Bold + Blue + '1. Thống kê mô tả độ dài tên sản phẩm:' + End)
print(lens.describe())
print(Bold+ Blue + '2. Độ lệch của phân phối (Skew):'+ End)
print(lens.skew())
print(Bold+ Blue + '3. Độ dốc của phân phối (Kurtosis):'+ End)
print(lens.kurtosis())


# $$\textbf{Nhận xét}$$
# 
# Từ Kết của thống kê trên, chúng ta có thể dễ dàng nhận thấy, độ dài chung trình của Tên sản phẩm là 22 Chữ. Trong đó, Tên Mặt hàng dài nhất gồm 217 Chữ. Ngược lại, Tên Sản phẩm ngắn nhất chỉ gồm 2 chữ. Mặt khác, chúng ta thấy, Sai số chuẩn cũng khá lớn, gần 13 Chữ. Đồng thời, độ dài của Tên sản phẩm phân phối không đều, có xu hướng nghiêng về bên trái (Hệ số Skew dương: 1.87), và dốc hơn so với Phân phổi chuẩn (Laplace Kurtosis: 8.35). Một cách trực quan hơn, chúng ta có thể dựng đồ thị Histogram như dưới đây
# 
# ##### Đồ thị phân bổ

# In[21]:


sns.set(color_codes=True)
plt.figure(figsize=(12, 4))
#lens.hist(bins=20,density=False)
sns.distplot(lens, kde=True, bins=50, color="steelblue")
plt.title('Sự Phân Phối Độ Dài Tên Của %d Sản Phẩm'%len(citigo_data['Tên sản phẩm'].unique()),
         fontsize=15, fontweight='bold')
plt.ylabel('Số lượng mặt hàng',fontsize=12)
plt.xlabel('Số lượng chữ trong Tên sản phẩm',fontsize=12)
plt.autoscale(enable=True, axis='both',tight=True)
plt.show()


# 
# 
# ## Đồ thị hoá
# ### Nhóm nhiều nhất
# #### Thống kê số lượng hàng

# In[22]:


depent_citigo_dummy=citigo_dummy.drop('Tên sản phẩm', axis=1)
label_count = depent_citigo_dummy.sum()
label_count_pan=label_count.reset_index() 
label_count_pan=label_count_pan.rename(columns={0: "Số mặt hàng"})
label_count_pan= label_count_pan.set_index(['index'])
label_count_pan.head(5)


# #### Đồ thị hoá
# ##### Không sắp xếp theo thứ tự 

# In[23]:


fig, ax=plt.subplots(figsize=(12,12)) 
plt.barh(label_count_pan['Số mặt hàng'].index,
        label_count_pan['Số mặt hàng'])
plt.autoscale(enable=True, axis='both',tight=True)
plt.title('Phân bổ của %d Mặt Hàng vào %d Nhóm Hàng Level 2 và 3'%(len(citigo_dummy['Tên sản phẩm'].unique()),
                                                                   len(label_count_pan)),
        fontsize=15, fontweight='bold')
plt.ylabel('Tên %d Nhóm hàng'%  len(label_count_pan),fontsize=12)
plt.xlabel('Số lượng Mặt Hàng', fontsize=12)
plt.grid(which='major',linestyle=':',linewidth=0.9)
for i,v in enumerate(label_count_pan['Số mặt hàng']):
    ax.text(v , i-0.15 , str(v), color='blue')


# ##### Sếp theo thứ tự giảm dần

# In[24]:


fig, ax=plt.subplots(figsize=(12,12)) 
plt.barh(label_count_pan['Số mặt hàng'].sort_values(ascending=True).index,
        label_count_pan['Số mặt hàng'].sort_values(ascending=True))
plt.autoscale(enable=True, axis='both',tight=True)
plt.title('Phân bổ của %d Mặt Hàng vào %d Nhóm Hàng Level 2 và 3'%(len(citigo_dummy['Tên sản phẩm'].unique()),
                                                                   len(label_count_pan)),
        fontsize=15, fontweight='bold')
plt.ylabel('Tên Nhóm hàng',fontsize=12)
plt.xlabel('Số lượng Mặt Hàng', fontsize=12)
plt.grid(which='major',linestyle=':',linewidth=0.9)
for i,v in enumerate(label_count_pan['Số mặt hàng'].sort_values(ascending=True)):
    ax.text(v , i-0.15 , str(v), color='blue')


# ### Phân bổ các Mặt Hàng theo Nhóm
# #### Đồ thị chung

# In[27]:


bars0=[]
for i,col in enumerate(depent_citigo_dummy.columns):
    sum_0=sum(depent_citigo_dummy[col]==0)
    bars0.append(sum_0)  

bars1=[]
for i,col in enumerate(depent_citigo_dummy.columns):
    sum_1=sum(depent_citigo_dummy[col]==1)
    bars1.append(sum_1)  
    
barWidth = 0.25
height=0.25 
r1 = np.arange(len(bars1))
r0 = [x + height for x in r1]
fig, ax=plt.subplots(figsize=(12,12)) 
plt.barh(r1,bars1, height,color='red',  label='Gía trị = 1')
plt.barh(r0,bars0, height,color='lightblue',  label='Giá trị = 0')
plt.title('Phân bổ của %d Mặt Hàng vào %d Nhóm Hàng Level 2 và 3'%(len(citigo_dummy['Tên sản phẩm'].unique()),
                                                                   len(label_count_pan)),
        fontsize=15, fontweight='bold',color='blue')
plt.ylabel('Tên %d Nhóm hàng'% len(label_count_pan),fontsize=12, fontweight='bold')
plt.xlabel('Số lượng Mặt Hàng', fontsize=12,fontweight='bold')
plt.yticks([r + barWidth for r in range(len(bars1))],
           depent_citigo_dummy.columns)
plt.grid(which='major',linestyle=':',linewidth=0.9)
plt.autoscale(enable=True, axis='both',tight=True)
plt.legend()
#plt.axis("off")


# $$\textbf{Nhận xét}$$
# 
# Chúng ta dễ dàng nhận thấy, sự bất cân đối trong phân phối giữa Giá trị 0 và 1 ở trong tất cả 45 Nhóm Hàng Level 2 và 3. Đặc biệt, Gía trị 0 chiếm đa phần. Phần Câu Lệnh dưới đây, sẽ tạo ra Đồ thị cho thấy sự Phân phối giữa Giá trị 0 và 1 trong từng Nhóm Hàng riêng rẽ
# #### Từng đồ thị phụ

# In[28]:


plt.figure(figsize=(15, 40))
for i,col in enumerate(citigo_dummy.loc[:,citigo_dummy.dtypes==np.uint8].columns):
    plt.subplot(15,3,i+1)
    sns.countplot(y=col,data=citigo_dummy.loc[:,citigo_dummy.dtypes==np.uint8],
              orient='h')


# ### Mối quan hệ giữa các Nhóm hàng
# Để có cái nhìn Tổng quan về mối quan hệ tuyến tính giữa 45 Nhóm Hàng phân theo tiêu chí Level 2 và 3, Chúng ta sẽ tính toán chỉ số Tương Quan Pearson giữa 
# #### Ma Trận Hệ số Tương Quan

# In[29]:


correlations=depent_citigo_dummy.corr()
plt.figure(figsize=(20, 15))
mask1 = np.zeros_like(correlations, dtype=np.bool)
mask1[np.triu_indices_from(mask1)] = True
cmap = 'Dark2'# sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlations,cmap=cmap, mask=mask1,annot=True,
            square=True
           ,vmax=.3, center=0,
            linewidths=.5, cbar_kws={"shrink": 0.7})
plt.title('Hệ số Tương Quan giữa %d Nhóm Hàng'% len(depent_citigo_dummy.columns),
          fontsize=15, fontweight='bold')
plt.ylabel('Tên %d Nhóm Hàng'%len(depent_citigo_dummy.columns),
          fontsize=13, fontweight='bold')
plt.ylabel('Tên %d Nhóm Hàng'%len(depent_citigo_dummy.columns),
          fontsize=13, fontweight='bold')
sns.set(font_scale=1)


# $$\textbf{Nhận xét:}$$
# 
# Mối quan hệ giữa các cặp Nhóm phần lớn là yếu, bởi Hệ số Tương quan giữa các Cặp tập trung trong khoảng, từ chưa đầy -0.14 tới 0. Chỉ một vài Cặp có Hệ số Tương Quan đáng chú ý. Để lọc ra Cặp Nhóm có Hệ số Tương Quan vượt qua một Ngưỡng (Threshold) nào đó, chúng ta thực hiện như sau
# 
# #### Hệ số Tương Quan vượt ngưỡng
# 
# Các bạn có thể tham khảo Chương trình Máy tính của mình, viết riêng cho mục đích này tại, Kho trên Github của mình
# 
# https://github.com/phuongvnguyen/Correlation-Analysis
# 
# ##### Xây dựng hàm

# In[30]:


def correlation_select(correlation, threshold):
    correlation_up=correlation.where(np.triu(np.ones(correlation.shape), k = 1).astype(np.bool))
    select_corr= [column for column in correlation_up.columns if any(abs(correlation_up[column])>threshold)]
    # printing
    print(Bold+ Red +'------------------------------------------------------------------'+End)
    print(Bold+ f'Cặp có Hệ số Tương Quan Vượt Ngưỡng {threshold}:'+End)
    print(len(select_corr))
    print(Bold+ Red +'------------------------------------------------------------------'+End)
    print(Bold+f'Danh Sách Cặp có Hệ số Tương Quan Vượt Ngưỡng {threshold}:' + End)
    print(select_corr)
    print(Bold+ Red +'------------------------------------------------------------------'+End)
    record_select_correlation=pd.DataFrame(columns=['Attribute_1','Attribute_2','Correlation_Value'])
    for column in select_corr:
        Attribute_11=list(correlation_up.index[abs(correlation_up[column])>threshold])
        Attribute_21=[column for _ in range(len(Attribute_11))]
        Correlation_Value1=list(correlation_up[column][abs(correlation_up[column])>threshold])
        temp_df_corr=pd.DataFrame.from_dict({'Attribute_1': Attribute_11,
                                      'Attribute_2': Attribute_21,
                                      'Correlation_Value': Correlation_Value1})
        record_select_correlation=record_select_correlation.append(temp_df_corr,ignore_index=True)
    print(Bold+f'Thống kê các Cặp có Hệ Số Tương Quan Vượt Ngưỡng {threshold}:')
    print(Bold+ Red +'------------------------------------------------------------------'+End)
    return record_select_correlation;


# ##### Khớp dữ liệu
# Chúng ta sẽ lọc ra các Cặp có Hệ số tương quan lớn hơn 0.5 về Giá trị tuyệt đối

# In[31]:


record_select_correlation=correlation_select(correlation=correlations, threshold=0.5)
record_select_correlation


# $$\textbf{Nhận xét:}$$
# 
# Từ kết quả trên, chúng ta rất dễ dàng nhận thấy như sau:
# 
# 1. Nếu một sản phẩm được phân loại vào Nhóm "Level 3_Trang điểm môi", thì nó cũng có khả năng cao sẽ thuộc vào Nhóm "Level 2_Trang điểm".
# 
# 2. Nếu một sản phẩm được phân loại vào Nhóm "Level 3_Đèn/máy xông tinh dầu spa và phụ kiện", , thì nó cũng có khả năng cao sẽ thuộc vào Nhóm "Level 2_Tinh dầu spa".
# 
# 3. Tương tự cho các trường hợp còn lại, etc
# 
# ### Tạo Đám Mây Chữ
# 
# 
# 
# Phần này Chúng ta Phân tích đặc điểm chữ, cụ thể Chúng ta hãy thử phân tích xem những Chữ nào quan trong nhất đóng vai là tín hiệu nhận dạng để phân phối vào các Nhóm Hàng Level 2 và Level 3. Để làm được điều này, trước tiên chúng ta hãy xây dựng một Hàm, giúp tạo ra Những Đám Mây Chữ (Word Clouds). 
# 
# #### Tạo hàm

# In[20]:


def myword_cloud(data,text,token,width,height,max_font_size, fig_size):
    """
    data: Dữ liệu đầu vào, cần tạo Đám Mây chữ
    text: Tên cột trong data ở dạng chữ, 
          ví dụ Cột "Tên sản phẩm", trong Dữ liệu "citigo_dummy"
    token: Tên cột mục tiêu trong data đầu vào.
    width: Chiều rộng Đám Mây Chữ, 1600
    height: Chiều cao Đám Mây Chữ, 800
    max_font_size: Cỡ chữ lớn nhất, 200
    fig_size: Thiết lập kích thước Đồ thị, (15, 7)
    """
    group=data[data[token]==1]
    group_text=group[text]
    neg_text=pd.Series(group_text).str.cat(sep='')
    myWordCloud=WordCloud(width=width, height=height,
                          max_font_size=max_font_size).generate(neg_text)
    plt.figure(figsize=fig_size)
    plt.imshow(myWordCloud.recolor(colormap="Blues"), interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Những Từ phổ biến nhất có liên quan tới Nhóm Hàng {token}",
              fontsize=20,fontweight='bold')
    plt.show()


# #### Khớp Dữ Liệu
# 
# ##### Một Nhóm Hàng

# In[21]:


myword_cloud(data=citigo_dummy,text='Tên sản phẩm',token='Level 3_Bộ trang điểm'
             ,width=1600,height=300,max_font_size=100, fig_size=(15,8))


# ##### Nhiều Nhóm Hàng
# ###### Ba Nhóm Cột đầu tiên

# In[36]:


for i,col in enumerate(depent_citigo_dummy.iloc[:,0:3].columns):
    myword_cloud(data=citigo_dummy,text='Tên sản phẩm',token=col
             ,width=1600,height=300,max_font_size=100, fig_size=(15,8))


# ###### Nhóm hàng Level 2
# Sáu Nhóm Cột cuối

# In[37]:


for i,col in enumerate(depent_citigo_dummy.columns[-6:]):
    myword_cloud(data=citigo_dummy,text='Tên sản phẩm',token=col
             ,width=1600,height=300,max_font_size=100, fig_size=(15,8))


# # Chuẩn bị Dữ Liệu Biến Đầu Vào và Đầu Ra
# 
# Chúng ta tiến hành phân tách dữ liệu thành hai nhóm: đầu ra và đầu vào. Biến đầu gồm 45 biến thuộc về Level 2 và Level 3, và chúng đã được biến đổi thành dạng số. Riêng biến đầu vào là "Tên sản phẩm" vẫn ở dạng Văn bản nên chúng ta, phải đổi chúng thành dạng số bằng cách tính
# chỉ số TF-IDF của chúng
# 
# ## Biến Đầu vào
# 
# ### Tính TF-IDF
# 
# Biến giải thích trong trường hợp này chính là "Tên sản phẩm". Tuy nhiên, để đào tạo một chương trình Máy học/AI mà có thể phân loại được sản phẩm từ tên của nó, trước tiên chúng ta phai chuyển đổi chúng thành dạng số. Dạng phổ biến nhất là $\textbf{TF-IDF}$ ($\textit{term frequency-inverse document frequency}$). Để hiểu rõ hơn về TF-IDF, có thể tham khảo thêm tại
# 
# http://www.tfidf.com/
# 
# Sau đây chúng ta sẽ tiến hành tính toán TF-IDF như là giá trị cho biến phụ thuộc "Tên sản phẩm". Chúng ta có thể tiến hành tính toán IF-IDF hoàn toàn bằng thủ công theo định nghĩa của nó tại
# 
# https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# 
# Các bạn có thể tham khảo quy trình Câu Lệnh thực hiện quá trình quy đổi Văn bản sang dạng IF-IDF trong một Dự án Kaggle về phát triển ML cho nhận dạng những Lời bình độc hại, của tôi trên Github
# 
# https://github.com/phuongvnguyen/Detecting-Toxic-Comments
# 
# Tuy nhiên, để nhanh chóng và thuận tiện chúng ta sẽ sử dụng hàm TfidfVectorizer như sau
# #### Chuẩn bị

# In[68]:


Name='Tên sản phẩm'


# import re, string
# re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
# def tokenize(s):
#     return re_tok.sub(r' \1 ', s).split()

# #### Khởi tạo hàm
# 
# Hàm TfidfVectorizer có nhiều hệ số chúng ta phải hiệu chỉnh, tuy nhiên để đơn giản hoá, trong trường hợp này tôi để tất cả cá hệ số ở dạng mặc định, trừ use_idf. Các hệ số khác của hàm này, có thể tham khảo tại 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# 

# In[23]:


tfidf_vectorizer = TfidfVectorizer(use_idf=True)
print(Bold+'Cầu Hình chi tiết của Hàm "tfidf_vectorizer()":'+End)
print(tfidf_vectorizer)


# #### Khớp dữ liệu

# In[24]:


tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(citigo_dummy[name])
tfidf_vectorizer_vectors


# $$\textbf{Nhận xét:}$$
# Như vậy chúng ta, có thể dễ dàng nhận thấy là cần 8,503 Chữ để đặt Tên cho 13,024 Mặt hàng

# #### Xem một vài Chữ
# 
# Thử xem 10 Chữ đầu tiên

# In[72]:


tfidf_vectorizer.get_feature_names()[0:10]


# Thử xem 10 Chữ cuối

# In[65]:


tfidf_vectorizer.get_feature_names()[-10:]


# #### Xem giá trị TF-IDF

# In[69]:


print(Bold+Blue+'Kich cỡ Ma trận tài liệu-từ (term-document Matrix):'+End)
print("Số Sản Phẩm: %d, Số Chữ: %d" % tfidf_vectorizer_vectors.shape)
print(Bold+Blue+'Tên sản phẩm đầu tiên:'+End)
print(citigo_dummy[Name][0])
print(Bold+Blue+'Giá trị TF-IDF của "Tên sản phẩm" đầu tiên:'+End)
print(tfidf_vectorizer_vectors[0])


# $$\textbf{Nhận xét:}$$
# Như vậy chúng ta, có thể dễ dàng nhận thấy là cần 8,503 Chữ Đơn để đặt Tên cho 13,024 Mặt hàng. Chẳng hạn, Chúng ta dễ dàng thấy rằng Tên sản phẩm đầu tiên là "Bạch Linh 02", bao gồm ba chữ "Bạch", "Linh", và "02". Vị trí của chúng trong Dữ liệu là:
# 
# - "02": ở vị trí 19;
# - "Bạch" ở vị trí 4613;
# - "Linh" ở vị trí 1869. 
# 
# Đồng thời,  Giá trị TF-IDF của chúng lần lượt là 0.53, 0.61, và 0.58. Về bản chất, thì giá trị TF-IDF này được tính toán bằng TF*IDF, trong đó giá trị TF (Term Frequency) đã được điều chỉnh bởi giá trị IDF (Inverse Document Frequency) của nó (term frequency is weighted by its IDF values).
# 
# 
# Và chúng ta lưu ý thêm một điều rằng
# 
# $\textbf{Term Frequency}$: Xác định vai trò của một Chữ đối với trong một Đoạn văn bản nhỏ/Câu cụ thể nào đó.
# 
# $$\color{red}{\textbf{The higher the TF value of a word, the more important it is to a given document}}$$
# 
# $\textbf{Inverse Document Frequency}$: Xác định vai trò của một Chữ đối với toàn bộ Văn Bản.
# 
# $$\color{red}{\textbf{The lower the IDF value of a word, the less unique it is to the whole collection of documents}}$$
# 
# 
# 
# ### Đặt biến đầu vào
# 

# In[28]:


X=tfidf_vectorizer_vectors
X


# ## Biến Đầu ra
# Để đơn gian, trong 45 Biến đầu ra, chúng ta chọn Biến đầu ra là 'Level 2_Chăm sóc cơ thể'

# In[29]:


Y=citigo_dummy['Level 2_Chăm sóc cơ thể']#['Level 3_Bộ trang điểm']
Y.head(3)


# $$\textbf{Nhận xét:}$$
# 
# Chúng ta đã có biến đầu vào là x có định dạng số TF-IDF, và 45 biến đầu ra ở dạng Binary, sẵn sàng cho đào tạo chương trình máy học tự dữ liệu đầu vào và đầu ra như trên. 
# 
# # Phân tách dữ liệu

# In[30]:


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                                                                test_size=validation_size,
                                                                random_state=seed)


# # Đào tạo Máy Học
# 
# Chúng ra biết rằng, có rất nhiều Thuật toán Máy học khác nhau và chúng ta không biết Thuật toán nào là phù hợp nhất với bộ dữ liệu của chúng ta. Do đó, bước đầu tiên chúng ta sẽ tiến hành đào tạo một loạt các Thuật Toán Máy Học khác nhau, rồi so sánh hiệu quả hoạt động của chúng. Trong trường hợp này, chúng ta sẽ dùng chỉ tiêu $\textbf{Chính xác (accuracy}$ làm tiêu chuẩn để đo lường sự hiệu quả trong hoạt động của các Máy Học. Để hiểu thêm về chỉ tiêu này cũng như các chỉ tiêu khác sử dụng trong đánh giá khả năng hoạt động của Máy học, các bạn có thể tham khảo thêm
# 
# https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
# 
# ## Chọn Thuật Toán
# ### Thiết lập lựa chọn
# Do sử dụng các Thuật toán cao cấp như Bagging, Boosting, và XGBoost, sẽ mất thời gian trên Máy tính cầu hình thấp, nên trong trường hợp này, chúng ta không dùng tới chúng. Tuy nhiên, nếu cấu hình Máy tính cao, các bạn có thể sử dụng hoặc sử dụng dịch vụ Máy Chủ của Amazon Web Servies (AWS). Lưu ý rằng, dịch vụ này AWS của Amazon được đánh giá tuyệt vời hơn các dịch vụ Máy chủ khác như Azure, nhưng không hề rẻ (90 Cent/giờ). Các bạn có thể tham khảo dịch vụ AWS của Amazon tại
# https://aws.amazon.com/

# In[31]:


models = []
models.append(('RidgeClass', RidgeClassifier()))
models.append(('LinearSVC', LinearSVC()))
models.append(('SGDClass', SGDClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NearCent',NearestCentroid()))
#models.append(('Percept', Perceptron()))
#models.append(('PassiveAgg', PassiveAggressiveClassifier()))
#models.append(('BerNB',BernoulliNB()))
#models.append(('ComNB',ComplementNB()))
#models.append(('MulNB',MultinomialNB()))
#models.append(('',))


# ### Đào tạo và Đánh giá
# 
# Qúa trình đào tạo và đánh các Thuật toán được thực hiên như sau. Mỗi chương trình Máy Học sẽ được đào tạo trong 10 lần (10 folds), sau đó giá trị trung bình về Độ chính xác sẽ được tính toán để xem Chương trình Máy học nào hoạt động tốt nhất. 
# 
# Thực chất, chúng ta sử dụng kỹ thuật $\textbf{K-Fold-Cross-Validation}$, để đào tạo và đánh giá khả năng hoạt động của các Chương Trình Máy học. $\textbf{K-Fold-Cross-Validation}$ là kỹ thuật lấy lại mẫu (resampling) phổ biến nhất. Nó cho phép chúng ta đào tạo và kiểm định Mô hình của chúng ta $\textbf{K lần}$ trên các Nhóm mẫu khác nhau của dữ liệu đào tạo (training data), và sau đó xây dựng một ước tính về hiệu xuất của một Mô hình Máy học trên dữ liệu không nhìn thấy (unseen data). 
# 
# 
# Bên cạnh giữ lại một dữ liệu kiểm định (a validation dataset) trong toàn bộ dữ liệu, thì $\textbf{K-Fold-Cross-Validation}$ là một trong hai kỹ thuật quan trọng để đào tạo Chương trình Máy học tránh khỏi tình trạng $\textbf{Quá Khớp (Overfitting)}$.
# 
# 
# Ngoài ra, các bạn có thể tham khảo thêm về vai trò của kỹ thuật $\textbf{K-Fold-Cross-Validation}$ tại
# 
# https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
# 
# https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/

# In[32]:


#print(Bold+'Kết quả so sánh Hoạt động của %d Chương Trình Máy Học:' %(len(models))+End)
competing_model_score_df = []
results = []
names = []
for name, model in models:
    scoring = 'accuracy'
    training_time = []
    kfold = KFold(n_splits=10, random_state=7)
    start = timer()
    cv_results = cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    training_time.append(timer() - start)
    val = [name, cv_results.mean(), cv_results.std(),sum(training_time)]
    competing_model_score_df.append(val)
    #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #print(msg) 


# In[33]:


compared_results = pd.DataFrame(competing_model_score_df,)
compared_results.columns = ['Mô hình', 'Chỉ số Accuracy Bình Quân',
                         'Sai Số Chuẩn', 'Thời gian đào tạo (s)']
print(Bold+'Kết quả so sánh hoạt động của %d Thuật Toán Máy Học khác nhau:'%(len(models))+End)
print(compared_results)


# ### Đồ thị hoá kết quả so sánh

# In[34]:


fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(111)
plt.boxplot(results,vert=False, showmeans=True)  
ax.set_yticklabels(names)
#for i,v in enumerate(cv_results.mean()):
 #   ax.text(v , i-0.15 , str(v), color='blue')#, fontweight='bold')
plt.autoscale(enable=True, axis='both',tight=True)
plt.grid(which='major',linestyle=':',linewidth=0.9)
plt.title('Kết quả so sánh hoạt động của %d Chương trình Máy Học khác nhau'%len(models),
          fontsize=14,fontweight='bold')
plt.ylabel('Tên %d Chương trình Máy Học'%(len(models)), fontsize=11)
plt.xlabel('Phân phối của Chỉ số Accuracy qua 10 Lần đào tạo lại', fontsize=11)
plt.show()


# $$\textbf{Nhận xét:}$$
# Dựa trên kết quả so sánh trên, chúng ta có thể nên chọn Chương trình Máy Học sử dụng Thuật toán Linear SVC (Linear Support Vector Classification). Bởi cho giá trị Chính xác cáo nhất và ổn định nhất. Chi tiết thuật toán này, các bạn có thể tham khảo thêm tại 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
# 
# https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989
# 
# https://medium.com/@ankitnitjsr13/math-behind-support-vector-machine-svm-5e7376d0ee4d
# 
# Tuy nhiên, lưu ý rằng, khi so sánh Hoạt Động của các Chương Trình Máy Học ở trên, thì tất các Hệ số trong các Thuật Toán đều được thiết lập ở dạng mặc định. Do đó, bước tiếp theo chung ta tiến hành cải thiện khả năng Hoạt động của Thuật toán Linear SVC, bằng cách tối ưu hoá cấu hình của nó
# 
# ## Tối ưu hoá hoạt động của Máy Học
# 
# ![OptLinSVC.png](attachment:OptLinSVC.png)
# 
# ### Thiết lập Mô Hình cần tối ưu hoá
# 
# 

# In[35]:


tuning_model = LinearSVC(dual=False,max_iter=3000)
tuning_model


# ### Thiết lập Vùng Lưới Tìm Kiếm
# Grid Search

# In[36]:


print(Bold+'Tiểu chuẩn sự dụng cho đánh giá:'+End)
penalty_values=['l1','l2']
print(penalty_values)
print(Bold+'Thông số C:'+End)
c_values=np.arange(0.1,2,0.1) # Có thể tuỳ chỉnh thêm nếu cần
print(c_values)
print(Bold+'Kết hợp:'+End)
param_grid =dict(penalty=penalty_values,C=c_values)
print(param_grid)


# ### Tối ưu hoá Mô Hình
# Sử dụng 10 Folds
# #### Thực hiện Tìm kiếm trên Lưới

# In[38]:


kfold = KFold(n_splits=10, random_state=1)
start = timer()
grid = GridSearchCV(estimator=tuning_model, param_grid=param_grid, 
                    scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print(Bold+"Kết quả Chính Xác Nhất: %f Với thiết lập %s" % (grid_result.best_score_,
                                                            grid_result.best_params_)+End)
print(Bold+"Time of seaching %.2fs" % (timer() - start))


# #### Hiển thị kết quả tìm kiếm trên Lưới

# In[39]:


print(Bold+'Kết quả tìm kiếm Cấu Hình Tối Ưu trên Lưới:\n'+End)
display(pd.DataFrame(grid_result.cv_results_)        .sort_values(by='rank_test_score').head(5))


# In[40]:


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# ## Hoàn thiện Mô Hình
# 
# ### Thiết lập lại Mô hình ở trạng thái tối ưu

# In[41]:


myML=LinearSVC(C=1.3,dual=False,penalty='l2',max_iter=3000)
print(Bold+ Blue+'Cấu hình của Chương trình Máy Học tối ưu:'+End)
print(myML)


# ### So sánh hoạt động với Mô Hình Cơ Bản
# #### Cấu hình Mô hình Cơ Bản và Tối Ưu

# In[42]:


base_LinearSVCmodel =LinearSVC(dual=False,max_iter=3000)
print(Bold+'Cấu Hình của Mô hình Cơ Bản:'+End)
print(base_LinearSVCmodel)
print(Bold+'Cấu Hình của Mô hình Tối Ưu:'+End)
print(myML)


# #### So Sánh Hoạt Động

# In[43]:


def evaluate(model, X_train, Y_train):
    trainedModel=model.fit(X_train, Y_train)
    predictions = trainedModel.predict(X_train)
    accuracy = accuracy_score(Y_train, predictions)
    return accuracy


# In[44]:


base_LinearSVCaccuracy = evaluate(base_LinearSVCmodel,X_train,Y_train)
Opt_LinearSVCaccuracy = evaluate(myML,X_train,Y_train)
print('Khả năng dự báo chính xác của Mô hình cơ bản là  {:0.4f}% và Mô hình Tối Ưu là {:0.4f}%'.      format(base_LinearSVCaccuracy,Opt_LinearSVCaccuracy))
print('Mô Hình Tối Ưu cải thiện được {:0.2f}% hoạt động so với Mô hình Cơ Bản'.      format( 100 * (Opt_LinearSVCaccuracy - base_LinearSVCaccuracy) / base_LinearSVCaccuracy))


# ### Đào tạo Mô Hình Tối Ưu

# In[45]:


myMLperform=myML.fit(X_train, Y_train)


# ### Lưu dữ Mô hình Tối Ưu

# In[46]:


filename = 'Phuong_Trained_Machine_Learning.sav'
dump(myMLperform, open(filename, 'wb'))


# ## Kiểm định hoạt động của Mô hình tối ưu
# 
# ### Tải Mô HÌnh Máy Học tối ưu đã đào tạo

# In[47]:


Phuong_TrainedML = load(open(filename, 'rb'))


# $$\textbf{Nhận xét:}$$
# 
# Tiếp theo chúng ta thử kiểm tra xem khả năng Học của Chương trình Máy Học này, xem liệu rằng việc thu thập Dữ liệu có giúp tăng khả năng Hoạt động của nó hay không. Để kiểm tra điều đó, chúng ta tiến hành xây dựng Đường Học (Learning Curves)
# 
# ### Kiểm tra khả năng học của Chương Trình Máy Học tối ưu
# 
# Để kiểm tra khả năng Học của một Chương trình Máy học, Đường Cong Học (Learning curve) là công cụ kiểm định được sử dụng rộng rãi. Theo đó, trong mỗi lượt thay đổi kích thước Dữ Liệu Đào tạo tăng dần, Chương Trình Máy học sẽ được đánh giá trên cả Dữ Liệu Đào tạo nó và Dữ liệu Kiểm định (Validation dataset). Đồng thời, Đồ thị đo lường hiệu xuất hoạt động của nó sẽ được tao ra để mô tả Đường Cong Học
# 
# Đường Học của một Chương Trình Máy Học trong quá trình đạo tạo nó có thể được sử dụng để kiểm tra các vấn đề liên quan, như
# 
# 1. Hiệu xuất hoạt động của Chương Trình qua thời gian.
# 
# 2. Thiếu Khớp (underfit) hoặc Quá Khơp (Overfit) của một Chương Trình Máy Học.
# 
# 3. Xem xét xem tỷ lệ phân chia giữa Dữ liệu đào tạo và kiểm định có phù hợp không.
# 
# 
# #### Thiết lập hàm
# 
# Chúng ta thiết kế Hàm để kiểm tra 6 trường hợp khác nhau, với tỷ lệ của Dữ liệu đào tạo tăng dần, lần lượt là 0.1, 0.2, 0.3, 0.6, 0.8, và 1.

# In[48]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, 
                        train_sizes=np.array([0.10,0.20, 0.30, 0.60, 0.80, 1. ])):
    """
    Hàm plot_learning_curve có 8 tham số đầu vào như sau
    1. estimator: Thuật toán Máy học cần kiểm tra 
    2. title: ở dạng Văn bản, để đặt tiêu đề cho đồ thị, ví dụ 'Đồ thì Đường Học'
    3. X: Biến đầu vào/giải thích.
    4. y: Biến đầu ra/phụ thuộc
    5. ylim: xác định giá trị nhỏ/lớn nhất của biến phụ thuộc được vẽ.
    6. cv: xác định các cách Phân tách Xác Nhận-Chéo (cross-validation splitting strategy).
       Có khả năng lựa chọn sau:
       - None, 5-fold cross-validation được sử dụng
       - Một số nguyên.
       - Kfold
    7. n_jobs: Khai báo số lượng chạy song song
    8. train_sizes: Khai báo các trường hợp kích thước của Dữ liệu để đào tạo Máy học.
    """
    #plt.figure()
    plt.figure(figsize=(12, 5))
    plt.suptitle(title,fontsize=12,fontweight='bold',color='blue')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Kích cỡ Mẫu đào tạo (Training data)")
    plt.ylabel("Điểm")

    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X, y, train_sizes=train_sizes,
                                                            cv=cv, n_jobs=n_jobs)
    # Tính giá trị bình quân
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Dựng khoảng tin cậy:
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.5,
                     color="y",label='Khoảng tin cậy trong 1 Đơn vị Sai Số Chuẩn tính trên dữ liệu Đào tạo')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="red",
                    label='Khoảng tin cậy trong 1 Đơn vị Sai Số Chuẩn tính trên kiểm định')
    # Dựng điểm trung bình
    plt.plot(train_sizes, train_scores_mean, marker='o', color="blue",
             label="Đường Cong Học tính trên dữ liệu đào tạo", linestyle='--')
    plt.plot(train_sizes, test_scores_mean, marker='v', color="green",
             label="Đường Cong Học tính trên dữ liệu kiểm tra")
    #plt.autoscale(enable=True, axis='both',tight=True)
    plt.grid(which='major',linestyle=':',linewidth=0.9)
    plt.legend(loc="best")
    return plt


# #### Mô tả Đường Học (Learning Curves)

# In[49]:


title = "Khả năng học của Chương Trình Máy Học Tối Ưu về nhận diện sản phẩm 'Level 2_Chăm sóc cơ thể'"
#kfold = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
kfold = KFold(n_splits=10, random_state=7)
estimator = Phuong_TrainedML
plot_learning_curve(estimator, title, X_train,Y_train,
                    ylim=(0.7, 1.01), cv=kfold, n_jobs=4)


# $$\textbf{Nhận xét:}$$
# 
# Hai điểm về khả năng học của Chương trình Máy học tối ưu (Linear SVC) về nhận diện Mặt hàng thuộc nhóm "Level 2_Chăm sóc cơ thể":
# 
# 1. Chương trình Máy Học càng trở nên "thông minh" hơn hay khả năng nhận diện sản phẩm thuộc nhóm hàng "Level 2_Chăm sóc cơ thể" tăng, khi dữ liệu dùng để đào tạo nó tăng lên. Tuy nhiên, khi kích thước Dữ liệu đào tạo tăng lên trên 6,000 Quan sát, thì Độ thông minh hay khả năng nhận diện sản phẩm của nó không tăng lên nữa.
# 
# 2. Sự sai khác giữa Độ chính xác trong nhận diện sản phẩm giữa Dữ liệu đào tạo và kiểm tra dần thu hẹp khi kích thước Dữ liệu đào tạo tăng lên. Cuối cùng, điểm số này giữa hai vùng Dữ liệu đều rất cao trên 95% và không sai khác nhiều. Điều đó có thể nói rằng, Chương trình Máy học của chúng ta là không bị tình trạng Underfit hoặc Overfit khi tỷ lệ phân chia giữa hai dữ liệu là 70/30 hoặc 80/20.
# 
# ### Thống kê kết quả kiểm định Mô hình Tối Ưu

# In[50]:


X_train


# In[51]:


X_validation


# In[52]:


PhuongML_predictionst=Phuong_TrainedML.predict(X_train)
PhuongML_predictions=Phuong_TrainedML.predict(X_validation)

print(Bold+'-------------KẾT QUẢ KIỂM TRA HOẠT ĐỘNG CỦA MÁY HỌC------------------'+End)
print(Bold+'I. Giá trị Chính xác: '+End)

print('1. Trên dữ liệu đào tạo: %.2f'%(accuracy_score(Y_train, PhuongML_predictionst)))

print('2. Trên dữ liệu kiểm định: %.2f'%(accuracy_score(Y_validation, PhuongML_predictions)))

print(Bold+'II. Ma Trận kiểm định (Confusion Matrix):'+End)
print('1. Trên dữ liệu đào tạo:')
print(classification_report(Y_train, PhuongML_predictionst))
print('2. Trên dữ liệu kiểm định:')
print(classification_report(Y_validation, PhuongML_predictions))
print(Bold+'---------------------------------------------------------------------'+End)
print(Bold+'BẠN HÀI LÒNG VỚI KHẢ NĂNG HOẠT ĐỘNG CỦA CHƯƠNG TRÌNH MÁY HỌC NÀY CHỨ?'+End)
print(Bold+'---------------------------------------------------------------------'+End)


# In[53]:


print(confusion_matrix(Y_validation, PhuongML_predictions))


# ### Dựng Đường Cong AUC-ROC
# #### Đánh giá hàm quyết định 
# ##### Cho mỗi giá trị trong dữ liệu đào tạo

# In[54]:


Ftrain_score=Phuong_TrainedML.decision_function(X_train)
Ftrain_score 


# ##### Cho mỗi giá trị trong dữ liệu kiểm định

# In[55]:


Ftest_score=Phuong_TrainedML.decision_function(X_validation)
Ftest_score 


# #### Tính Đường Cong và diện tích ROC
# ##### Cho dữ liệu đào tạo

# In[56]:


#y_train=np.asarray()
fpr_train, tpr_train, thresholds_train=roc_curve(Y_train, Ftrain_score)
roc_auc_train=auc(fpr_train, tpr_train)
plt.figure(figsize=(12, 5))
plt.plot(fpr_train, tpr_train, color='darkorange',
          label='Đường Cong ROC (Diện tích = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.grid(which='major',linestyle=':',linewidth=0.9)
plt.title('Đường cong ROC và Phần diện tích dưới nó (AUC) trên dữ liệu train',
          fontsize=15,fontweight='bold')
#plt.autoscale(enable=True, axis='both',tight=True)
plt.xlabel('Tỷ lệ FPR', fontsize=10)
plt.ylabel('Tỷ lệ TPR',fontsize=10)
plt.legend(loc="lower right")
#plt.axis('off')


# ##### Cho dữ liệu kiểm định

# In[57]:


fpr_test, tpr_test, thresholds_test=roc_curve(Y_validation, Ftest_score)
roc_auc_test=auc(fpr_test, tpr_test)
plt.figure(figsize=(12, 5))
plt.plot(fpr_test, tpr_test, color='darkorange',
          label='Đường Cong ROC (Diện tích = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.grid(which='major',linestyle=':',linewidth=0.9)
plt.title('Đường cong ROC và Phần diện tích dưới nó (AUC) trên dữ liệu test',
          fontsize=15,fontweight='bold')
#plt.autoscale(enable=True, axis='both',tight=True)
plt.xlabel('Tỷ lệ FPR', fontsize=10)
plt.ylabel('Tỷ lệ TPR',fontsize=10)
plt.legend(loc="lower right")


# #### AUC-ROC của cả dữ liệu train và test 

# In[58]:


plt.figure(figsize=(12, 5))
plt.plot(fpr_train, tpr_train, color='darkorange',
          label='Đường Cong ROC của dữ liệu Train (Diện tích = %0.4f)' % roc_auc_train,
        linestyle='-')
plt.plot(fpr_test, tpr_test, color='b',
          label='Đường Cong ROC dữ liệu Test (Diện tích = %0.4f)' % roc_auc_test,
        linestyle='-.')
plt.plot([0, 1], [0, 1], color='navy', linestyle='-')
plt.grid(which='major',linestyle=':',linewidth=0.9)
plt.title('Đường cong ROC và Phần diện tích dưới nó (AUC)',
          fontsize=15,fontweight='bold')
#plt.autoscale(enable=True, axis='both',tight=True)
plt.xlabel('Tỷ lệ FPR', fontsize=10)
plt.ylabel('Tỷ lệ TPR',fontsize=10)
plt.legend(loc="lower right")


# ### Những Từ, Chương Trình Máy Học không thể phân loại

# In[573]:


Phuong_TrainedML


# In[ ]:


PhuongML_predictions=Phuong_TrainedML.predict(X_validation)


# In[574]:


PhuongML_predictions


# In[ ]:


test_combined = pd.concat([test, test_y], axis=1)


# In[576]:


Y_validation


# In[577]:


X_validation


# In[579]:


#test_combined = pd.concat([PhuongML_predictions, Y_validation], axis=1)
test_combined=pd.DataFrame({'Giá trị dự báo':PhuongML_predictions,
                            'Giá trị kiểm định thực':Y_validation})
test_combined


# In[583]:


value10=len(test_combined[(test_combined['Giá trị dự báo'] == 1) & (
    test_combined['Giá trị kiểm định thực'] == 0)])


# In[584]:


value01=len(test_combined[(test_combined['Giá trị dự báo'] == 0) & (
    test_combined['Giá trị kiểm định thực'] == 1)])


# In[592]:


print('Số Lần Dự báo sai: {}'.format(value10+value01))
#value10+value01
#print(Bold + Blue + 'Số quan sát: {}'.format(len(citigo_data)))


# ## Cải thiện hoạt động của Máy học
# 
# Chúng ta sẽ sử dụng Các Mô hình Nhóm (Ensemble Models), để cải thiện hoạt động của Chương trình Máy Học, bởi các Mô Hình Nhóm là sự kết hợp của một vài Mô HÌnh, cho phép tạo ra các hoạt động dự báo tốt hơn so với một Mô hình đơn. Do đó, trong phần này chúng ta sẽ sử dụng Mô Hình Nhóm để kiểm tra xem liệu điều đó có thực sự giúp cải thiện hoạt động của Chương Trình Máy học của chúng ta.
# 
# Có nhiều dạng của Mô Hình Nhóm, bao gồm cả Bagging và Boosting. Do đó, trước tiên chúng ta sẽ tiến hành so sánh các Dạng Mô Hình Nhóm với nhau. Tuy nhiên, sử dụng Dữ Liệu để đào tạo các Thuật Toán liên quan tới Mô Hình Nhóm, thường yêu cầu Máy tính có cầu hình đủ mạnh, nếu không sẽ mất thời gian để đào tạo chúng. Do đó, trong phần này, chúng ta chỉ sử dụng hai Thuật Toán của Mô Hình Kết Hợp, như là một ví dụ để kiểm tra xem liệu việc sử dụng chúng có làm nâng cao khả năng hoạt động của Chương trình Máy Học của chúng ta hay không.
# 
# ### Sử dụng Thuật Toán Ensemble
# 
# 
# #### Lựa chọn Thuật Toán
# 
# ##### Thiết lập lựa chọn

# In[170]:


ensem_models = []
ensem_models.append(('RanForest', RandomForestClassifier(n_estimators=100)))
#ensem_models.append(('ExtTree',ExtraTreesClassifier(n_estimators=100)))
ensem_models.append(('AdaBoost',AdaBoostClassifier()))
#ensem_models.append(('GradientBoost',GradientBoostingClassifier()))
#models.append(('XgbBoost',XGBClassifier()))
#models.append(('',))


# ##### Đào tạo và Đánh giá

# In[103]:


ensem_competing_model_score = []
ensem_results = []
for name, model in ensem_models:
    ensem_names = []
    ensem_scoring = 'accuracy'
    ensem_training_time = []
    kfold = KFold(n_splits=10, random_state=7)
    start = timer()
    cv_results = cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
    ensem_results.append(cv_results)
    ensem_names.append(name)
    training_time.append(timer() - start)
    val = [name, cv_results.mean(), cv_results.std(),sum(training_time)]
    ensem_competing_model_score.append(val)


# In[104]:


compared_ensem_results = pd.DataFrame(ensem_competing_model_score,)
compared_ensem_results.columns = ['Mô hình', 'Chỉ số Accuracy Bình Quân',
                         'Sai Số Chuẩn', 'Thời gian đào tạo']
compared_ensem_results


# #### Tối ưu hoá Thuật Toán
# 
# ##### Thiệt lập Mô hình cần tối ưu hoá

# In[189]:


tuning_RFC_model=RandomForestClassifier(n_estimators=100)
print(Bold+ 'Cấu hình hiện tại của Mô Hình:'+End)
pprint(tuning_RFC_model.get_params())
print(Bold+ 'Số lượng Tham Số trong Thuật Toán:'+End)
print(len(tuning_RFC_model.get_params()))


# 
# 
# 
# ##### Thiết lập Vùng Lưới Tìm Kiếm 
# 
# Để hiểu rõ hơn bản chất của Thuật Toán RandomForest, và cách tối ưu hoá, chúng ta có thể tham khảo thêm tại
# 
# Bài Nghiên cứu Lý Thuyết gốc xây dựng Thuật Toán này tại 
# 
# https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
# 
# Từ lý thuyết tới Thực hành
# 
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 
# Từ trên, chúng ta thấy, để tối ưu hoá Thuật Toán RandomForest, có Hai hệ số quan trọng nhất mà chúng ta cần chọn, đó là Sô Cây trong Rừng (n_estimators), và Số Đặc điểm (max_features) được tính đến cho phân tách ở mỗi Nút Lá (leaf node). Tuy nhiên, Chúng ta sẽ thử tối ưu hoá Thuật Toán này bằng cách lựa chọn một loạt các tham số dưới đây
# 
# 1. n_estimators = number of trees in the foreset
# 2. max_features = max number of features considered for splitting a node
# 3. max_depth = max number of levels in each decision tree
# 4. min_samples_split = min number of data points placed in a node before the node is split
# 5. min_samples_leaf = min number of data points allowed in a leaf node
# 6. bootstrap = method for sampling data points (with or without replacement)
# 
# 
# Mặt khác, quá trình thực hiện Tối ưu hoá Thuật toán Random Forest nói riêng và các Thuật Toán Máy Học khác nói chung, chúng ta có hai Phương Pháp.
# 
# 1. Tạo Lưới tìm Ngẫu Nhiên 
# 
# 2. Tìm Lưới (Grid Searching)
# 
# Để thấy rõ hơn hai phương pháp này, các bạn có thể tham khảo Dự Án của tôi về giới thiệu quy trình tối ưu hoá Thuật toán Random Forest tại Kho Github của tôi
# 
# https://github.com/phuongvnguyen/Optimizing-Random-Forest-Algorithm
# 
# Trong dự án này, để đơn gian tôi chỉ sử dụng Phương Pháp Grid Searching mà thôi. Thêm nữa, do Cấu hình máy tính cá nhân có hạn, nên tổi chỉ thực hiện quá trình Tối ưu hoá Thuật Toán này ở dạng đơn giản. Tuy nhiên, với Máy tính có cấu hình cao hoặc sử dụng Máy Chủ của Amazon Web Services (AWS) thì toàn bộ quy trình nên thực hiện đầy đủ, sẽ tạo kết qua tốt hơn
# ##### Thiết lập Lưới ngẫu nhiên
# 
# Chúng ta sẽ tiến hành so sánh lựa chọn một loạt các kết hợp cài đặt Cấu hình nhau. 

# In[215]:


# Số lượng Cây trong thiết lập Khu Rừng Ngẫu Nhiên
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 250, num = 2)]

# Số lượng Đặc điểm để xem xét tại mỗi Phân Tách
max_features = ['auto', 'sqrt']

# Số lượng Tầng tối đa trong Cây
max_depth = [int(x) for x in np.linspace(10, 200, num = 2)]
max_depth.append(None)

# Số lượng mẫu nhỏ nhất cần có để phân tách một Nốt
min_samples_split = [2, 5]

# Số lượng mẫu cần có tại mỗi Nút Lá
min_samples_leaf = [1, 2]

# Phương pháp chọn mẫu cho đào tạo mỗi Cây.
bootstrap = [True, False]

# Tổng hợp các thiết lập riêng lẻ để Tạo Lưới Ngẫu Nhiên
grid_param_searching = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(Bold+'Tập hợp các Tham Số có Trong Lưới để lựa chọn:'+End)
pprint(grid_param_searching)

print(Bold+'Số lượng Cấu Hình được thiết lập dựa trên lựa chọn Tham số từ Lưới:'+End)
print(len(grid_param_searching['bootstrap'])*
      len(grid_param_searching['max_depth'])*
      len(grid_param_searching['max_features'])*
      len(grid_param_searching['min_samples_leaf'])*
      len(grid_param_searching['min_samples_split'])*
      len(grid_param_searching['n_estimators']))


# $$\textbf{Nhận xét:}$$
# 
# Như vậy, chúng ta sẽ tiến hành lựa chọn và so sánh 4320 Cấu Hình khác nhau của Thuật Toán Khu Rừng Ngẫu Nhiên (Random Forest). Tuy nhiên, nếu chỉ sử dụng Tìm kiếm ngẫu nhiên trên Lưới (Random Grid Search) thì lợi ích của việc tìm kiếm Ngẫu nhiên này đó là Chúng ta sẽ không thực hiện hêt mọi kết hợp, mà chỉ lựa chọn ngẫu nhiên để lẫy mẫu một loạt các giá trị mà thôi. Tuy nhiên, như vậy mục đích sẽ không đạt được là tìm kiếm Thuật toán tối ưu, nên chúng ta không dùng kỹ thuật đó mà sẽ tìm kiếm và so sánh tất cả các khả năng xảy ra. Tuy nhiên, điều này tốn thời gian và cần có Máy Tính Cấu Hình cao.

# ##### Thiết lập K-Fold-Cross-Validation cho Grid-Searching 

# In[216]:


#kfold = KFold(n_splits=10, random_state=1)
grid_searching_RFC = GridSearchCV(estimator = tuning_RFC_model, 
                              param_grid = grid_param_searching, 
                          cv = 3, n_jobs = -1, verbose = 2, 
                              return_train_score=True)
print(grid_searching_RFC)


# ##### Thực hiện tìm kiếm trên Lưới

# In[217]:


start = timer()
# Fit the grid search to the data
RFC_Grid_res=grid_searching_RFC.fit(X_train, Y_train);
print(Bold+"The Best Model: %f With the configuration %s" % (RFC_Grid_res.best_score_, 
                                                    RFC_Grid_res.best_params_)+End)
print(Bold+"Thời gian tìm kiếm %.2fs" % (timer() - start))


# ##### Hiển thị kết quả tìm kiếm trên Lưới
# 
# ###### Hiển thị toàn bộ kết quả

# In[255]:


print(Bold+'Kết quả lựa chọn từ Lưới:'+End)
display(pd.DataFrame(RFC_Grid_res.cv_results_)        .sort_values(by='rank_test_score').head(5))


# ###### Hiện thị vài lựa chọn

# In[237]:


print(Bold+'Kết quả lựa chọn (một phần) từ Lưới:'+End)
display(pd.DataFrame(RFC_Grid_res.cv_results_)[['mean_test_score',
                                                'std_test_score','params',
                                               'rank_test_score']].
        sort_values(by='rank_test_score').head(5))


# #### Xác lập Cấu hình tối ưu RFC

# In[295]:


Opt_RFC=RandomForestClassifier(bootstrap=True,max_depth=None,max_features='auto',
                              min_samples_leaf=1,min_samples_split=2,n_estimators=200)
pprint(Opt_RFC)


# #### So sánh hiệu quả với Chương Trình ban đầu

# In[305]:


Opt_RFCaccuracy=evaluate(Opt_RFC,X_train,Y_train)
print('Chỉ số Accuracy của Mô hình RFC Tối Ưu: {:0.4f}%'.format(100*Opt_RFCaccuracy))


# In[303]:


Opt_LinearSVCaccuracy


# In[301]:


print('Mô Hình Tối Ưu cải thiện được {:0.2f}% hoạt động so với Mô hình Cơ Bản'.      format( 100 * (Opt_RFCaccuracy-Opt_LinearSVCaccuracy)             / Opt_LinearSVCaccuracy))


# ### Sử dụng phương thức Voting
# 
# #### Tập hợp các Thuật Toán
# Do ba Thuật Toán, RidgeClass, LinearSVC, và SGDClass, có khả năng dự báo gần giống nhau, nên chúng ta kết hợp ba Thuật Toán này xem có giúp cải thiện khả năng dư báo của Máy Học không? Ngoài ra, với Máy Tính có Cấu Hình cao thì nên thử cả với sự kết hợp với các Thuật Toán Ensemble

# In[267]:


clf1=RidgeClassifier()
clf2=LinearSVC()
clf3=SGDClassifier()
collect=[('RidgeClass',clf1),
         ('LinearSVC',clf2),
         ('SGDClass',clf3)]
print(Bold+'Danh sách tập hợp các Thuật Toán:'+End)
pprint(collect)


# #### Thiết lập cấu hình

# In[265]:


voting_clf = VotingClassifier(estimators=collect,voting='hard')
print(Bold+'Cấu hình kết hợp:'+End)
pprint(voting_clf)


# #### Đào tạo và đánh giá Voting

# In[333]:


#kfold = KFold(n_splits=10, random_state=7)
start = timer()
trained_voting_cvresults = cross_val_score(voting_clf, X_train,Y_train, cv=kfold,
                                          scoring = 'accuracy',n_jobs=-1)
time=timer() - start
print(Bold+'Kết quả đánh giá Mô hình với chiến lược Voting theo %d lượt đào tạo'      %(len(trained_voting_cvresults))+End)
print(pd.DataFrame({'Tên':['Voting'] ,
              'Số lượt đào tạo':[len(trained_voting_cvresults)],
              'Accuracy TB':[trained_voting_cvresults.mean()],
             'Sai số chuẩn':[trained_voting_cvresults.std()],
             'Thời gian đào tạo':[time]}))


# $$\textbf{Nhận xét:}$$
# Chiến lượng Voting này cũng tạo ra một kết quả rất tốt, Nhưng vẫn chưa tạo được sự khác biệt vượt trội so với Chương Trình Máy học LinearSVC. Tuy nhiên, lưu ý rằng tỷ trọng kết hợp ba Thuật Toán được thiết lập ở chế độ Mặc định, chưa phải là tỷ trọng tối ưu. Do đó, tỷ trọng tối ưu cần được tìm trước khi so sánh với với Chương Trình Máy học LinearSVC, là cần thiết. Dòng Lệnh dưới đây thực hiện một tìm kiếm minh hoạ
# 
# #### Tìm kiếm tỷ trọng kết hợp tối ưu

# In[381]:


result_weiSearch=pd.DataFrame(columns=('w1', 'w2', 'w3', 'Accurac TB',
                                       'Sai số chuẩn','Thời gian'))
i = 0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):
            voting_weight_clf=VotingClassifier(estimators=collect,voting='hard',
                                            weights=[w1,w2,w3])
            start = timer()
            trained_voting_weicvresults = cross_val_score(voting_weight_clf, X_train,
                                                              Y_train,cv=kfold, 
                                                              scoring = 'accuracy',n_jobs=-1)
            time=timer() - start
            result_weiSearch.loc[i] = [w1, w2, w3, trained_voting_weicvresults.mean(),
                                       trained_voting_weicvresults.std(),time]
            i += 1
print(Bold+'Kết quả tìm kiếm Tỷ trọng kết hợp tối ưu cho Voting:'+End)
print(Bold+'----------Kết quả xếp theo thứ tự giảm dần:---------'+End)
print(result_weiSearch.sort_values(by=['Accurac TB'],ascending=False))                  


# $$\textbf{Nhận xét:}$$
# Chúng ta có thể nhận thấy, với Tỷ trọng tối ưu (1,3,1), cho kết quả dự báo khá là cạnh tranh so với Mô hình LinearSVC. Tuy nhiên, vùng tím kiếm nên được mở rộng, và sự kết hợp các Thuật Toán nên được đa dạng hơn

# w1 = w2 = w3 = range(1, 4)
# w123 = list(itertools.product(w1, w2, w3))
# len(w123)
# 
# print('Tỷ Trọng: {}'.format(w123[1]))
# print('Tỷ Trọng: {}'.format(w123[2]))
# 
# AccMean=list()
# AccStd=list()
# Weight=list()
# Time=list()
# print(Bold + 'Tìm kiếm Tỷ Trọng tối ưu:'+ End)
# for param in w123:
#     voting_weight_clf=VotingClassifier(estimators=collect,voting='hard',
#                                             weights=[w1,w2,w3])
#     start = timer()
#     trained_voting_weicvresults = cross_val_score(voting_weight_clf, X_train,
#                                                               Y_train,cv=kfold, 
#                                                   scoring = 'accuracy')                                                         
#     time=timer() - start
#     AccMean.append(trained_voting_weicvresults.mean())
#     AccStd.append(trained_voting_weicvresults.std())
#     Weight.append(param)
#     Time.append(time)  
#     
# weight_search=pd.DataFrame({'Tỷ trọng':Weight,'Accuracy TB':AccMean,
#                         'Sai số chuẩn':AccStd, 'Thời gian':Time}).sort_values(by=['Accuracy TB'],ascending=False)
# weight_search

# ### Phát triển và đào tạo AI

# #### Xử lý dữ liệu cho AI
# ##### Thiết lập Chỉ Số cho các Chữ trong Tên SP

# In[116]:


max_features=30000
# Thiết lập Hàm
tokenizer = Tokenizer(num_words=max_features)
# cap Nhật từ vựng
tokenizer.fit_on_texts(list(citigo_dummy[Name]))
# Liệt kê Chỉ Số cho Chữ
list_tokenized_word = tokenizer.texts_to_sequences(citigo_dummy[Name])
print(Bold+'Danh sách Tên của %d Sản phẩm đầu:'%       (len(citigo_dummy[Name].head(5)))+End)
print(citigo_dummy[Name].head(5))
print(Bold+'Chỉ số của các Chữ trong Tên của %d Sản phẩm đầu:'%       (len(list_tokenized_word[:5]))+End)
print(list_tokenized_word[:5])


# ##### Độ dài/Số Chữ trong mỗi tên SP

# In[117]:


totalNumWords = [len(one_product) for one_product in list_tokenized_word]
print(Bold+'Chỉ số của các Chữ trong Tên của %d Sản phẩm đầu:'%       (len(totalNumWords[:5]))+End)
print(totalNumWords[:5])


# ##### Phân bổ Độ dài tên SP

# In[118]:


sns.set(color_codes=True)
plt.figure(figsize=(12, 4))
#lens.hist(bins=20,density=False)
sns.distplot(totalNumWords, kde=False, bins=20, color="steelblue")
plt.title('Sự Phân Phối Độ Dài Tên Của %d Sản Phẩm'%len(citigo_data['Tên sản phẩm'].unique()),
         fontsize=15, fontweight='bold')
plt.ylabel('Số lượng mặt hàng',fontsize=12)
plt.xlabel('Số lượng chữ trong Tên sản phẩm',fontsize=12)
plt.autoscale(enable=True, axis='both',tight=True)
plt.show()


# $$\textbf{Nhận xét:}$$
# Do độ dài\Số Chữ trong các sản phẩm là khác nhau, chúng ta đồng bộ hoá chiều dài/số lượng Chữ trong mỗi tên sản Phẩm. Cụ thể Độ dài/Số Chữ sẽ có độ dài là 15 Chữ
# ##### Padding và Dữ liệu đầu vào

# In[189]:


maxlen =15
X_ai = pad_sequences(list_tokenized_word,maxlen=maxlen)
X_ai[:1]


# ###### Dữ liệu đầu ra Level 2

# In[278]:


# Đầu ra cho Level 2
list_level2=citigo_dummy.iloc[:,-6:].columns
list_level2
Y_ai_level2=citigo_dummy[list_level2].values #['Level 2_Chăm sóc cơ thể']
#Y_ai=np.asarray(Y_ai)
Y_ai_level2


# ###### Dữ liệu đầu ra Level 3

# In[282]:


list_level3=citigo_dummy.iloc[:,1:40].columns
list_level3
Y_ai_level3=citigo_dummy[list_level3].values 
Y_ai_level3


# ##### Phân tách dữ liệu 

# In[283]:


validation_size = 0.20
seed = 7
X_ai_train_level2, X_ai_validation_level2,Y_ai_train_level2, Y_ai_validation_level2 = train_test_split(X_ai, Y_ai_level2,
                                                                test_size=validation_size,
                                                                random_state=seed)
X_ai_train_level3, X_ai_validation_level3,Y_ai_train_level3, Y_ai_validation_level3 = train_test_split(X_ai, Y_ai_level3,
                                                                test_size=validation_size,
                                                                random_state=seed)


# #### Thiết lập Cấu Hình cho AI
# ![Screenshot%202020-03-05%2011.03.50.png](attachment:Screenshot%202020-03-05%2011.03.50.png)
# ##### Thiết lập cấu hình
# ###### Level 2

# In[288]:


inp = Input(shape=(maxlen,))
x2 = Embedding(max_features, embed_size)(inp)
x2 = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x2)
x2 = GlobalMaxPool1D()(x2)
x2 = Dense(50, activation="relu")(x2)
x2 = Dropout(0.1)(x2)
x2 = Dense(6, activation="sigmoid")(x2)# bao nhieu bien dau ra
myAI_Level2 = Model(inputs=inp, outputs=x2)
myAI_Level2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ###### Level 3

# In[290]:


x3 = Embedding(max_features, embed_size)(inp)
x3 = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x3)
x3 = GlobalMaxPool1D()(x3)
x3 = Dense(50, activation="relu")(x3)
x3 = Dropout(0.1)(x3)
x3 = Dense(39, activation="sigmoid")(x3)# bao nhieu bien dau ra
myAI_Level3 = Model(inputs=inp, outputs=x3)
myAI_Level3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ##### Đào tạo
# ###### Level 2

# In[289]:


print(Bold+ Underline+'Kích Cỡ Dữ Liệu:'+End)
print(Bold+'1. Đào tạo:'+End)
print('Đầu vào: %d. Đầu ra: %d'%(len(X_ai_train_level2),len(Y_ai_train_level2)))
print(Bold+'2. Kiểm định:'+End)
print('Đầu vào: %d. Đầu ra: %d'%(len(X_ai_validation_level2),len(Y_ai_validation_level2)))
print(Bold+ Underline+'Quá trình đào tạo AI:'+End)
start = timer()
myTrainedAI_Level2=myAI_Level2.fit(X_ai_train_level2, Y_ai_train_level2, batch_size=32, 
                             epochs=10, validation_split=0.3);
#myAI=model.fit(X_ai_train, Y_ai_train, batch_size=32, epochs=10,validation_split=0.0,
 #         validation_data=(X_ai_validation,Y_ai_validation));
print(Bold+"Thời gian đào tạo AI cho Level 2 %.2fs" % (timer() - start))  
print(myAI_Level2.summary())


# In[295]:


print(Bold+ Underline+'Kích Cỡ Dữ Liệu:'+End)
print(Bold+'1. Đào tạo:'+End)
print('Đầu vào: %d. Đầu ra: %d'%(len(X_ai_train_level3),len(Y_ai_train_level3)))
print(Bold+'2. Kiểm định:'+End)
print('Đầu vào: %d. Đầu ra: %d'%(len(X_ai_validation_level3),len(Y_ai_validation_level3)))
print(Bold+ Underline+'Quá trình đào tạo AI:'+End)
start = timer()
myTrainedAI_Level3=myAI_Level3.fit(X_ai_train_level3, Y_ai_train_level3, batch_size=32, 
                             epochs=10, validation_split=0.3);
#myAI=model.fit(X_ai_train, Y_ai_train, batch_size=32, epochs=10,validation_split=0.0,
 #         validation_data=(X_ai_validation,Y_ai_validation));
print(Bold+"Thời gian đào tạo AI cho Level 2 %.2fs" % (timer() - start))  
print(myAI_Level3.summary())


# ##### Kiểm định AI

# In[320]:


print(Bold+'1. Kết quả kiểm định AI cho dự báo Level 2'+ End)
accr_level2 = myAI_Level2.evaluate(X_ai_validation_level2 ,Y_ai_validation_level2 )
print('Dữ Liệu kiểm định (Level 2)\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.      format(accr_level2[0],accr_level2[1]))
print(Bold+'2. Kết quả kiểm định AI cho dự báo Level 3'+ End)
accr_level3 = myAI_Level3.evaluate(X_ai_validation_level3 ,Y_ai_validation_level3 )
print('Dữ Liệu kiểm định (Level 3)\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.       format(accr_level3[0],accr_level3[1]))


# ##### Lưu giữ AI

# In[322]:


AI_level2 = 'Phuong_Trained_AI_Level2.sav'
dump(myAI_Level2, open(AI_level2, 'wb'))


# In[324]:


AI_level3 = 'Phuong_Trained_AI_Level3.sav'
dump(myAI_Level3, open(AI_level3, 'wb'))


# ##### Thử AI

# In[323]:


Phuong_TrainedAI2 = load(open(AI_level2, 'rb'))


# In[325]:


Phuong_TrainedAI3 = load(open(AI_level3, 'rb'))


# In[294]:


list_level2=citigo_dummy.iloc[:,-6:].columns
#list_level2
list_level3=citigo_dummy.iloc[:,0:40].columns
list_level3=list_level3[1:40]
#pprint(list_level3)


# In[328]:


txt = ["Vacosy set mắt 4 ô"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=maxlen)
predict_level2 = Phuong_TrainedAI2.predict(padded)
predict_level3= Phuong_TrainedAI3.predict(padded)
print('Sản phẩm:', txt)
print('Thuộc vào Nhóm: \n')
print( list_level2[np.argmax(predict_level2 )])
print('\n Và Nhóm: \n')
print(list_level3[np.argmax(predict_level3 )])


# # Kết thúc
# 
# Sau khi sử dụng và so sánh một loạt các Mô hình khác nhau, thì AI nên được sử dụng bởi không chỉ hiệu xuất hoạt động mà còn quá trình đào tạo cũng đơn giản hơn nhiều so với các Mô hình Máy học khác, đặc biệt Mô hình liên quan tới các Thuật Toán Ensemble và Voting. Tuy nhiên, AI vẫn còn tồn tại và cách khắc phục để cải thiện thêm như sau:
# 
# 1. AI mới chỉ sử dụng các Thiết Lập cấu hình đơn giản, chưa ở dạng tối ưu. Do đó, tối ưu hoá AI là cần thiết (hyperparameter tunning). Để làm điều này, chúng ta sẽ sử dụng Kỹ thuật tìm kiếm trên lưới (Grid Search).
# 
# 2. Mạng lưới Thần Kinh (Neural Networks) còn nhiều Thuật Toán khác, chẳng hạn. Tuy nhiên, trong trường hợp này, chỉ có Long Short-Term Memory (LSTM), thuộc nhóm recurrent neural network (RNN), được sử dụng. Do đó, các Thuật Toán khác liên quan tới Neural Network, như  Gated recurrent units (GRUs), hay CNNs, etc, nên được đào tạo và sử dụng thêm để kiểm tra.
# 
# 3. AI cần được kiểm định thêm về khả năng học, và tình trạng overfitting hoặc underfitting, etc
# 
# 4. Bộ dữ liệu thô có sự mất cân đối cực lớn, những Nhóm sản phẩm có ít Mặt Hàng, chẳng hạn như Nước hoa, hay Tinh dầu spa, chỉ thì AI rất khó có thể đào tạo để nhận biết những sản phẩm này
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

# In[ ]:




