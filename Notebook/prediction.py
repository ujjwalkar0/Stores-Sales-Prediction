import pickle as pk
import pandas as pd

with open('rfr.pk', 'rb') as f:
    rfr = pk.load(f)

with open('normal_Item_MRP.pk', 'rb') as f:
    normal_Item_MRP = pk.load(f)

with open('standard_Item_Visibility.pk', 'rb') as f:
    standard_Item_Visibility = pk.load(f)

with open('standard_Item_Weight.pk', 'rb') as f:
    standard_Item_Weight = pk.load(f)

def predict(**kwargs):
    df = pd.DataFrame(kwargs)
    df['Item_Visibility'] = standard_Item_Visibility.transform(df[['Item_Visibility']])
    df['Item_MRP'] = normal_Item_MRP.transform(df[['Item_MRP']])
    df['Item_Weight'] = standard_Item_Weight.transform(df[['Item_Weight']])
    
    print(rfr.predict(df))

predict(
    Item_Weight = [1.8676258373898538],
    Item_Fat_Content = [0],
    Item_Visibility = [-4.033762474581528],
    Item_Type = [13],
    Item_MRP = [0.3250115450699156],
    Outlet_Size = [1],
    Outlet_Location_Type = [0],
    Outlet_Type = [1],
    Years_Established = [21]
)