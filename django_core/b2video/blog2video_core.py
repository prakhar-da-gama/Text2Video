

from . import config
title = config.title
blog = config.blog
print(blog)

import os
home_path = os.getcwd()
home_path = home_path+"\b2video"
print(home_path)
print("--------------------------")
from moviepy.editor import *
import nltk
import string
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import itertools
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
import keybert
from keybert import KeyBERT
from nltk.corpus import wordnet
from moviepy.editor import AudioClip

# importing libraries
import os
import cv2
from PIL import Image
import os
import cv2
from PIL import Image
import os
import shutil
import os
import shutil

import os
import sys

import urllib
import zipfile


from skimage import io

from bing_image_downloader import downloader
from tkinter.constants import N
from gtts import gTTS
from moviepy.editor import VideoFileClip, concatenate_videoclips
import nltk
"""
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase




def Text_preprocess(text):
    
    #Decontraction
    pre_sample_text=decontracted(text)
   
    
    #remove words with numbers python
    #pre_sample_text = re.sub("\S*\d\S*", "",pre_sample_text).strip()
    
    #remove special character
    pre_sample_text= re.sub('[^A-Za-z0-9]+', ' ',pre_sample_text)
    
    
    #Removing stopwords
    stopwords= set(['the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
    pre_sample_text = ' '.join(e for e in pre_sample_text.split() if e.lower() not in stopwords)

    #Chunking
    chunked = ne_chunk(pos_tag(word_tokenize(pre_sample_text)))
    Tokens=[ w[0] if isinstance(w, tuple) else "_".join(t[0] for t in w) for w in chunked ]
    pre_sample_text=' '.join(Tokens)

    #Lowercase
    pre_sample_text=pre_sample_text.lower()

    return pre_sample_text

# Keyword Finder returns the top keywords and top keyphrases

# phrases_maxlen=set the maximum length of the phrases

# n_keywords=set the number of keywords required


def Keyword_Finder(doc,phrases_maxlen,n_keywords):
    
    keywords_list=[]
    keyphrases_list=[]
    
    pre_processed_doc=Text_preprocess(doc)
    
    kw_model = KeyBERT()
    print('Highlighted Keywords\n')
    keywords = kw_model.extract_keywords(pre_processed_doc,top_n=n_keywords,diversity=0.7,use_maxsum=True,highlight=True)
    keywords_list.append(keywords)
    print('Highlighted Keyphrases\n')
    phrases=keywords = kw_model.extract_keywords(pre_processed_doc,keyphrase_ngram_range=(1,phrases_maxlen),highlight=True,top_n=n_keywords,use_maxsum=True,diversity=0.7)
    keyphrases_list.append(phrases)
    return keywords_list,keyphrases_list


doc = blog

len(doc.split(" "))

doc = blog
#len(doc.split(" "))
phrases_maxlen=2
n_keywords=10

keywords,keyphrases=Keyword_Finder(doc,phrases_maxlen,n_keywords)

words=[]
for k,v in keywords[0]:
    words.append(k)
key_image=' '.join(words)
print(key_image)
print(words)
"""
words = ["Hello"]


#Converting text to audio 

# Import the required module for text
# to speech conversion


# This module is imported so that we can
# play the converted audio
import os

# The text that you want to convert to audio
mytext = blog


# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False,tld='co.in')

# Saving the converted audio in a mp3 file named
# welcome
myobj.save("AudioClip.mp3")

# Loading audioclip and getting its duration
audioclip =AudioFileClip("AudioClip.mp3")
audioclip_duration=audioclip.duration
print(audioclip_duration)








#Image generation parameters !
n = 3
print(audioclip_duration)
size = len(words)

number_images = int(audioclip_duration/n)
if number_images > len(words):
  number_images = len(words)
print(number_images)


for x in words:
  downloader.download(x, limit=1,  output_dir='dataset', adult_filter_off = True, force_replace=False, timeout=60)



#Number of Images we have
NumberOfImages = 0
for x in words:
  img_path = os.getcwd()+"/content/dataset/"+x+"/Image_1.jpg"
  try:
    image = io.imread(img_path)
    NumberOfImages = NumberOfImages + 1
  except:
    print(">")
print(NumberOfImages)

Frame_Per_Second = 0
if audioclip_duration != 0:
  #
  Frame_Per_Second = NumberOfImages/audioclip_duration
print(Frame_Per_Second)


counter = 0;
for x in words:
  try:

  
    os.rename(os.getcwd()+"/content/dataset/"+x+"/Image_1.jpg", "/content/dataset/"+x+"/"+x+".jpg")
    #os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    #src_path = r"/content/dataset/"+x+"/"+x+".jpg"
    #dst_path = r"/content/images/"+x+"/"+x+".jpg"
    #shutil.move(src_path, dst_path)
  except:
    print("no")


for x in words:
  try:

  
    
    #os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    src_path = r""+os.getcwd()+"/content/dataset/"+x+"/"+x+".jpg"
    dst_path = r""+os.getcwd()+"/content/images/"+x+".jpg"
    shutil.move(src_path, dst_path)
  except:
    print("no")

import os
import shutil
counter = 0;
for x in words:
  counter = counter + 1;
  try:

  
    os.rename(os.getcwd()+"/content/images/"+x+".jpg", "./content/images/"+str(counter)+".jpg")
    
  except:
    print("no")

# importing libraries


# Checking the current directory path
print(os.getcwd())

# Folder which contains all the images
# from which video is to be generated
#os.chdir("C:\\Python\\Geekfolder2")
path = os.getcwd()+"/content/images"

mean_height = 0
mean_width = 0

num_of_images = len(os.listdir(path))
print(num_of_images)

for file in os.listdir(path):
	im = Image.open(os.path.join(path, file))
	width, height = im.size
	mean_width += width
	mean_height += height
	# im.show() # uncomment this for displaying the image

# Finding the mean height and width of all images.
# This is required because the video frame needs
# to be set with same width and height. Otherwise
# images not equal to that width height will not get
# embedded into the video
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)


# Checking the current directory path
print(os.getcwd())

# Folder which contains all the images
# from which video is to be generated
os.chdir(os.getcwd()+"/content/images")
path = os.getcwd()+"/content/images"

mean_height = 0
mean_width = 0

num_of_images = len(os.listdir('.'))
# print(num_of_images)

for file in os.listdir('.'):
	im = Image.open(os.path.join(path, file))
	width, height = im.size
	mean_width += width
	mean_height += height
	# im.show() # uncomment this for displaying the image

# Finding the mean height and width of all images.
# This is required because the video frame needs
# to be set with same width and height. Otherwise
# images not equal to that width height will not get
# embedded into the video
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)

# print(mean_height)
# print(mean_width)

# Resizing of the images to give
# them same width and height
for file in os.listdir('.'):
	if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
		# opening image using PIL Image
		im = Image.open(os.path.join(path, file))

		# im.size includes the height and width of image
		width, height = im.size
		print(width, height)

		# resizing
		imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
		imResize.save( file, 'JPEG', quality = 95) # setting quality
		# printing each resized image name
		print(im.filename.split('\\')[-1], " is resized")


# Video Generating function
def generate_video():
	image_folder = '.' # make sure to use your folder
	video_name = 'mygeneratedvideo.avi'
	os.chdir(os.getcwd()+"/content/images")
	
	images = [img for img in os.listdir(image_folder)
			if img.endswith(".jpg") or
				img.endswith(".jpeg") or
				img.endswith("png")]
	
	# Array images should only consider
	# the image files ignoring others if any
	print(images)

	frame = cv2.imread(os.path.join(image_folder, images[0]))

	# setting the frame width, height width
	# the width, height of first image
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, 0, Frame_Per_Second, (width, height))

	# Appending the images to the video one by one
	for image in images:
		video.write(cv2.imread(os.path.join(image_folder, image)))
	
	# Deallocating memories taken for window creation
	cv2.destroyAllWindows()
	video.release() # releasing the video generated


# Calling the generate_video function
generate_video()



# Import everything needed to edit video clips
#Combining the audio and the Images to make a video clip


# loading video dsa gfg intro video
clip = VideoFileClip("mygeneratedvideo.avi")


# getting only first 5 seconds
#clip = clip.subclip(0, 5)

# loading audio file
audioclip = AudioFileClip(os.getcwd()+"/content/AudioClip.mp3")

# adding audio to the video clip
video_clip = clip.set_audio(audioclip)

video_clip.duration = audioclip.duration
# set the FPS to 1
video_clip.fps = 1
# write the resuling video clip
video_clip.write_videofile("Video_with_audio.mp4")

# showing video clip
video_clip.ipython_display()

clip_1 = VideoFileClip(os.getcwd()+"/content/images/Video_with_audio.mp4")
clip_2 = VideoFileClip(os.getcwd()+"/sequence_02_5.mp4")
final_clip = concatenate_videoclips([clip_1,clip_2])
final_clip.write_videofile(os.getcwd()+"/content/images/finall.mp4")


