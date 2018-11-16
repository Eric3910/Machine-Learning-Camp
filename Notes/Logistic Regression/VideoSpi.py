import requests
import re
from lxml import etree

def fetchUrl(url):
    '''
    Func: fetchUrl is used to get html page content
    Parameter:
    url : the url for the website 
    return : req.test as page content
    '''
    try:
        #imitate browser
        headers={'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8','User_Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36'
}
        #get page content 
        req = requests.get(url,headers = headers)
        # check request status for throw exception
        req.raise_for_status() 
        #encoding the content based on the response from the url
        req.encoding=req.apparent_encoding 
        print('Fetch URL successful.')
        return req.text
    except requests.RequestError as e:
        print(e) #throw error
    except Exception as e:
        print(e) #throw error

def parseHtml(page,links):
    '''
    Func: parseHtml is used to parse HTML Doc.
    Parameters：
        page: page content
        urating:container for parse result
    '''
	try:
		html = etree.HTML(page)
  		result= etree.tostring(html)
  		print(result.decode('utf-8'))

    except Exception as e:
        print(e) 