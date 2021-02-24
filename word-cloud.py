import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import random

# function to return random values in the blue color range at HSL format
def blue_color_func(**kwargs):
    return "hsl(238, 100%%, %d%%)" % random.randint(15, 80)


# get data directory
d = os.path.dirname(__file__)

# load our book
text = open(os.path.join(d, 'Lord of the Flies.txt'), encoding="utf-8").read()

# load image. This has been modified in GIMP to be brighter and have more saturation.
image_color = np.array(Image.open(
    os.path.join(d, "pig.png")))
# subsample by factor of 3. Very lossy but for a wordcloud we don't really care.
image_color = image_color[::3, ::3]

# adding specific stopwords
stopwords = set(STOPWORDS)
stopwords.add("ll")
stopwords.add("back")
stopwords.add("away")
stopwords.add("don't")
stopwords.add("dont")
stopwords.add("don")
stopwords.add("t")
stopwords.add("s")
stopwords.add("came")
stopwords.add("said")
stopwords.add("ve")
stopwords.add("go")
stopwords.add("got")
stopwords.add("now")
stopwords.add("one")
stopwords.add("come")
stopwords.add("looked")
stopwords.add("went")

# create mask
lord_mask = image_color.copy()
lord_mask[lord_mask.sum(axis=2) == 0] = 255

# some finesse: we enforce boundaries between colors so they get less washed out.
# For that we do some edge detection in the image
edges = np.mean([gaussian_gradient_magnitude(
    image_color[:, :, i] / 255., 2) for i in range(3)], axis=0)
lord_mask[edges > .08] = 255

# create default wordcloud
wcd = WordCloud(background_color="white", max_words=2000, stopwords=stopwords,
                max_font_size=120, margin=4, width=800, height=800, normalize_plurals=True)

# create masked wordcloud
wc = WordCloud(background_color="black", max_words=2000, mask=lord_mask, stopwords=stopwords,
               max_font_size=70, relative_scaling=0, margin=1, normalize_plurals=True)

# generate word cloud
wc.generate(text)
wc.to_file("masked_with_default_colors.png")
wcd.generate(text)
wcd.to_file("default_word_cloud.png")

# create custom colors for our word clouds
image_colors = ImageColorGenerator(image_color)
wc.recolor(color_func=image_colors)
wc.to_file("masked_with_image_colors.png")

wcd.recolor(color_func=blue_color_func, random_state=3)
wcd.to_file("default_word_cloud_with_blue_colors.png")
