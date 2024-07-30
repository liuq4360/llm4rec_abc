import csv
import json


def get_metadata_dict(path: str = '../../data/amazon_review/beauty/meta_All_Beauty.json') -> dict:
    item_dict = {}  # meta_All_Beauty.json中获取每个item对应的信息
    """
    {"category": [], "tech1": "", "description": ["Start Up combines citrus essential oils with gentle Alpha Hydroxy Acids to cleanse and refresh your face. The 5% AHA level is gentle enough for all skin types.", "", ""], 
    "fit": "", "title": "Kiss My Face Exfoliating Face Wash Start Up, 4 Fluid Ounce", 
    "also_buy": ["B000Z96JDI", "B00006IGL8", "B007C5X34G", "B00006IGLF", "B00213WCNC", "B00D1W1QXE", "B001FB5HZG", "B000FQ86RI", "B0012BSKBM", "B0085EVLRO", "B00A2EXVQE"], 
    "tech2": "", "brand": "Kiss My Face", "feature": [], "rank": [], 
    "also_view": ["B000Z96JDI", "B001FB5HZG", "B00213WCNC", "B00BBFOVO4", "B0085EVLRO"], 
    "details": {"\n    Product Dimensions: \n    ": "2.5 x 1.6 x 7 inches ; 4 ounces", "Shipping Weight:": "4 ounces", "ASIN: ": "B00006IGL2", "UPC:": "890795851488 701320351987 601669038184 793379218755 028367831938 787734768894 756769626417", "Item model number:": "1200040"},
     "main_cat": "All Beauty", "similar_item": "", "date": "", "price": "", 
     "asin": "B00006IGL2", "imageURL": ["https://images-na.ssl-images-amazon.com/images/I/41i07fBAznL._SS40_.jpg", 
     "https://images-na.ssl-images-amazon.com/images/I/31W8DZRVD1L._SS40_.jpg"], "imageURLHighRes": ["https://images-na.ssl-images-amazon.com/images/I/41i07fBAznL.jpg", "https://images-na.ssl-images-amazon.com/images/I/31W8DZRVD1L.jpg"]}
    
    """
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter='\n')
        for row in reader:
            j = json.loads(row[0])
            item_id = j['asin']
            title = j['title']
            brand = j['brand']
            description = j['description']
            price = j['price']
            also_buy = j['also_buy']
            also_view = j['also_view']
            if price != "" and '$' not in price and len(price) > 10:  # 处理一些异常数据情况
                price = ""
            item_info = {
                "title": title,
                "brand": brand,
                "description": description,
                "also_buy": also_buy,
                "also_view": also_view,
                "price": price
            }
            item_dict[item_id] = item_info
    return item_dict
