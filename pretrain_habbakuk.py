import os
from PIL import Image, ImageFont, ImageDraw
#creates path if folder habbakuk doesnt exist
if not os.path.exists('C:/Users/MANUEL/Desktop/handwritten/habbakuk/'):
        os.makedirs('C:/Users/MANUEL/Desktop/handwritten/habbakuk/')
        
#path to pretrain
path_pretrain_habbakuk="C:/Users/MANUEL/Desktop/handwritten/habbakuk"
#Load the font and set the font size to 42
font = ImageFont.truetype('C:/Users/MANUEL/Desktop/handwritten/Habbakuk.ttf', 42)

#Character mapping for each of the 27 tokens
char_map = {'Alef' : ')', 
            'Ayin' : '(', 
            'Bet' : 'b', 
            'Dalet' : 'd', 
            'Gimel' : 'g', 
            'He' : 'x', 
            'Het' : 'h', 
            'Kaf' : 'k', 
            'Kaf-final' : '\\', 
            'Lamed' : 'l', 
            'Mem' : '{', 
            'Mem-medial' : 'm', 
            'Nun-final' : '}', 
            'Nun-medial' : 'n', 
            'Pe' : 'p', 
            'Pe-final' : 'v', 
            'Qof' : 'q', 
            'Resh' : 'r', 
            'Samekh' : 's', 
            'Shin' : '$', 
            'Taw' : 't', 
            'Tet' : '+', 
            'Tsadi-final' : 'j', 
            'Tsadi-medial' : 'c', 
            'Waw' : 'w', 
            'Yod' : 'y', 
            'Zayin' : 'z'}

#Returns a grayscale image based on specified label of img_size
def create_image(label, img_size):
    if (label not in char_map):
        raise KeyError('Unknown label!')

    #Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)    
    draw = ImageDraw.Draw(img)

    #Get size of the font and draw the token in the center of the blank image
    w,h = font.getsize(char_map[label])
    draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), char_map[label], 0, font)

    return img

#Creates a 50x50 images of all the hebrew alphabet and save it to disk
#To get the raw data cast it to a numpy array

for character in list(char_map.keys()):
    #creates individual characters
    img = create_image(character, (50, 50)) 
    #stores characters into folders
    os.mkdir(os.path.join(path, character)) 
    img.save(f'C:/Users/MANUEL/Desktop/handwritten/habbakuk/{character}/example_{character}.png')
