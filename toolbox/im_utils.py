"""Provide serveral Image Utility functions.

This module allows the user to use several of the most commonly used image handling functions at Miro AI

Examples:
	>>> from toolbox.im_utils import get_pil_im
	>>> get_pil_im("https://random.imagecdn.app/500/150").size
	(500, 150)

This module contains the following functions:

- `get_pil_im(fp_url_nparray)`- Returns a PIL Image object from either a file path, URL, or numpy array
"""

import os, sys, time, json, warnings, io, validators
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps, ImageColor
import numpy as np
from typing import Union, Any, Tuple, List

def get_pil_im(fp_url_nparray: Union[str,np.ndarray,Image.Image]) -> Union[Image.Image, None]:
	''' return a PIL image object

	Args:
		fp_url_nparray: filepath, URL, numpy array, or even an PIL Image object

	Returns:
		A PIL Image object

	Raises:
		ValueError: An error when fp_url_nparray cannot be converted into an image

	Examples:
		>>> get_pil_im("https://random.imagecdn.app/500/150").size
		(500, 150)
	'''
	import urllib.request as urllib
	from urllib.error import HTTPError, URLError

	im = fp_url_nparray
	pil_im = None
	if isinstance(im, Image.Image):
		pil_im = im
	elif type(im) == np.ndarray:
		pil_im = Image.fromarray(im)
	elif os.path.isfile(im):
		pil_im = Image.open(im)
	elif validators.url(im):
		try:
			r = urllib.Request(im, headers = {'User-Agent': "Miro's Magic Image Broswer"})
			con = urllib.urlopen(r)
			pil_im = Image.open(io.BytesIO(con.read()))
		except HTTPError as e:
			warnings.warn(f'get_pil_im: error getting {im}\n{e}')
		except URLError as e:
			warnings.warn(f'get_pil_im: URL error using {im}\n{e}')
	else:
		raise ValueError(f'get_im: im must be np array, filename, or url')
	return pil_im

def propose_wh(im: Any , max_long_edge: int = None, min_short_edge: int = None) -> Tuple[int,int]:
	''' returns a Tuple of Width and Height keeping the aspect ratio

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		max_long_edge: max long edge of the resized image
		min_short_edge: min short edge of the resize image

	Returns:
		A Tuple of resized Width Hieght (keeping aspect ratio)

	Raise:
		AssertionError: Either max_long_edge or min_short_edge must be provided but not both

	Examples:
		>>> propose_wh("https://random.imagecdn.app/500/150", max_long_edge = 250)
		(250, 75)
		>>> propose_wh("https://random.imagecdn.app/500/150", min_short_edge = 100)
		(500, 150)
	'''
	assert any([max_long_edge, min_short_edge]) and not(all([max_long_edge, min_short_edge])), \
		f"Either max_long_edge or min_short_edge must be provided but not both"

	w_ , h_ = get_pil_im(im).size
	wh_ratio = w_/h_
	if w_ > h_:
		if max_long_edge:
			w = max_long_edge if w_ > max_long_edge else w_
			h = int(w/wh_ratio)
		else:
			h = min_short_edge if h_ < min_short_edge else h_
			w = int(h * wh_ratio)
	else:
		if max_long_edge:
			h = max_long_edge if h_ > max_long_edge else h_
			w = int(h * wh_ratio)
		else:
			w = min_short_edge if w_ < min_short_edge else w_
			h = int(w/wh_ratio)
	return int(w), int(h)

def get_im(fp_url_nparray: Union[str,np.ndarray,Image.Image], b_pil_im: bool = False) -> Union[Image.Image, np.ndarray, None]:
	''' returns an image (wrapper around get_pil_im)

	Args:
		fp_url_nparray: filepath, URL, numpy array, or even an PIL Image object
		b_pil_im: if True, returns a PIL Image Object

	Returns:
		An Image represented either in numpy array or as an PIL Image object

	Raises:
		ValueError: An error when fp_url_nparray cannot be converted into an image

	Examples:
		>>> get_im("https://random.imagecdn.app/500/150").shape
		(150, 500, 3)
	'''
	pil_im = get_pil_im(fp_url_nparray)
	return pil_im if b_pil_im else np.array(pil_im)

def im_crop(im: Any, x0: int, y0: int, x1:int, y1: int) -> np.ndarray:
	''' crops an image (numpy array) given a bounding box coordinates

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		x0: top left x value
		y0: top left y value
		x1: bottom right x value
		y1: bottom right y value

	Returns:
		An Image representation in an numpy array

	Raises:
		AssertionError: An error will be raised for nonsensical x or y values

	Examples:
		>>> im = get_im("https://random.imagecdn.app/500/150")
		>>> bbox = {'x0': 100, 'y0': 50, 'x1': 200, 'y1': 100}
		>>> im_crop(im, **bbox).shape
		(50, 100, 3)
	'''
	im_rgb_array = get_im(im)
	assert x1> x0, f'x1 ({x1}) cannot be greater than x0 ({x0})'
	assert y1> y0, f'y1 ({y1}) cannot be greater than y0 ({y0})'
	h,w,c = im_rgb_array.shape
	assert h>y0 and w>x0, f'x0 ({x0}) or y0 ({y0}) number out of bound w: {w}; h: {h}'
	return im_rgb_array[int(y0): int(y1), int(x0):int(x1), :]

def im_isgray(im: Any, b_check_rgb: bool = False) -> bool:
	''' [check if an image is grayscale](https://stackoverflow.com/a/58791118/14285096)

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		b_check_rgb: check all the pixels if the image has 3 channels

	Returns:
		True or False

	Examples:
		>>> im = get_im("https://picsum.photos/200/300?grayscale")
		>>> im_isgray(im, b_check_rgb = False)
		True
		>>> im_isgray("https://picsum.photos/200/300", b_check_rgb = True)
		False
	'''
	im_rgb_array = get_im(im)
	if len(im_rgb_array.shape) < 3: return True
	if im_rgb_array.shape[2]  == 1: return True
	if b_check_rgb:
		r, g, b = im_rgb_array[:,:,0], im_rgb_array[:,:,1], im_rgb_array[:,:,2]
		if (b==g).all() and (b==r).all(): return True
	return False

def pil_im_gray2rgb(pil_im: Image.Image) -> Image.Image:
	''' convert a grayscale Pillow Image to RGB format

	Args:
		pil_im: a PIL Image Object

	Returns:
		A PIL Image Object in RGB format

	Examples:
		>>> pil_im = get_pil_im("https://picsum.photos/200/300?grayscale")
		>>> pil_im = pil_im_gray2rgb(pil_im)
		>>> im_isgray(np.array(pil_im), b_check_rgb = False)
		False
	'''
	return pil_im.copy().convert('RGB'
			) if im_isgray( np.array(pil_im), b_check_rgb = False) else pil_im

def im_3c(im:Any)-> np.ndarray:
	'''returns a 3-channel (rgb) image in numpy array

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath

	Returns:
		a numpy array of shape: (h, w, 3)

	Examples:
		>>> im = get_im("https://picsum.photos/200/300?grayscale")
		>>> im_3c(im).shape
		(300, 200, 3)
	'''
	im_rgb_array = get_im(im)
	if im_isgray(im_rgb_array, b_check_rgb = False) :
		im = pil_im_gray2rgb(get_pil_im(im_rgb_array))
		im_rgb_array = np.array(im)
	return im_rgb_array[:,:,:3]

def im_min_short_edge(im: Any, short_edge: int,
		resample_algo: int = Image.LANCZOS) -> np.ndarray:
	''' Return an image whose short edge is no shorter than the given size

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		resample_algo: default to LANCZOS because it gives [the best downscaling quality](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table)

	Returns:
		a numpy array

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> im_min_short_edge(im, short_edge = 300).shape
		(450, 300, 3)
	'''
	w, h = propose_wh(im, min_short_edge = short_edge)
	im = get_pil_im(im).resize(size =(w,h), resample = resample_algo)
	return np.array(im)

def im_max_long_edge(im: Any, long_edge: int,
		resample_algo: int = Image.LANCZOS) -> np.ndarray:
	''' Return an image whose long edge is no longer than the given size

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		resample_algo: default to LANCZOS because it gives [the best downscaling quality](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table)

	Returns:
		a numpy array

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> im_max_long_edge(im, long_edge = 500).shape
		(300, 200, 3)
	'''
	w, h = propose_wh(im, max_long_edge = long_edge)
	im = get_pil_im(im).resize(size =(w,h), resample = resample_algo)
	return np.array(im)

def im_add_alpha(im: Any) -> np.ndarray:
	''' add an **empty alpha channel** for a  3-channel image

	for adding an alpha channel with an actual mask, see `im_apply_mask`

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath

	Returns:
		a numpy array of shape: (h, w, 4)

	Raises:
		AssertionError: An error will be raise if numpy array is not of shape (h,w,3)

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> im_add_alpha(im).shape
		(300, 200, 4)
	'''
	im_rgb_array = get_im(im)
	assert len(im_rgb_array.shape) > 2, f'im_rgb_array shape is only length {len(im_rgb_array.shape)}'
	h,w,c = im_rgb_array.shape
	assert c == 3, f'im_rgb_array mmust be 3-channel, found {c} channels'
	return np.concatenate((
			im_rgb_array, np.ones((h, w, 1))+254
			), axis =2).astype('uint8')

def im_to_bytes(im: Any, format: str = None, quality: Union[int, str] = "keep") -> bytes:
	''' Returns an Image representation in bytes format

	This function creates a mutable bytearray object and uses the `PIL.Image.save()` function
	to output the image to it. Note that the `PIL.Image.tobytes()` is [not recommended](https://stackoverflow.com/a/58949303/14285096).
	For more on working with binary data, see [here](https://www.devdungeon.com/content/working-binary-data-python#video_bytes_bytearray).

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		format: image format as [supported by PIL](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)
		quality: compression quality passed to `Image.save()`, defaults to "keep" (retain image quality, only for JPEG format); more details available [here](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg-saving)

	Returns:
		a bytes representation of the provide image

	Examples:
		>>> im_bytes = im_to_bytes("https://picsum.photos/200/300")
		>>> type(im_bytes)== bytes
		True
	'''
	imgByteArr = io.BytesIO()
	pil_im = get_pil_im(im)

	# from the PIL doc:
	# If a file object was used instead of a filename, this parameter should always be used.
	img_format = format if format else pil_im.format
	img_format = img_format if img_format else 'jpeg'
	pil_im.save(imgByteArr, format = img_format, quality = quality)
	imgByteArr = imgByteArr.getvalue()
	return imgByteArr

def simple_im_diff(im_1: Any, im_2: Any, n_pixels_allowed: int = 0) -> bool:
	''' compare two images and check for pixel differences [using PIL](https://stackoverflow.com/a/73433784/14285096)

	Args:
		im_1: first image representation in numpy array, PIL Image object, URL, or filepath
		im_2: second image representation in numpy array, PIL Image object, URL, or filepath
		n_pixels_allowed: threshold to number of pixel difference allowed between im_1 and im_2

	Returns:
		True if two images are different (above `n_pixels_allowed`) and False otherwise

	Examples:
		>>> simple_im_diff("https://picsum.photos/200/300","https://picsum.photos/200/300")
		True
	'''
	if get_im(im_1).shape != get_im(im_2).shape: return True

	from PIL import ImageChops
	im_1, im_2 = get_pil_im(im_1), get_pil_im(im_2)
	diff = ImageChops.difference(im_1,im_2)
	channels = diff.split()
	for c in channels:
		if len(set(c.getdata()))> n_pixels_allowed: return True
	return False

def im_write_text(im: Any, text: str, x:int, y:int,
		tup_font_rgb: Tuple[int,int,int] = (255,255,255),
		tup_bg_rgb: Union[Tuple[int,int,int], None] = None
	) -> np.ndarray:
	''' draw text over image and returns it as a numpy array

	font_size is not currently supported because PIL's default font
	[cannot change in size](https://github.com/python-pillow/Pillow/issues/6622);
	for simplicity, the work-arounds (such as [this](https://stackoverflow.com/a/48381516/14285096))
	is not implemented.

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		text: text to be written
		x: text's top-left x location
		y: text's top-left y location
		tup_font_rgb: RGB values for font color, defaults to white
		tup_bg_rgb: RGB values for text's background color

	Returns:
		a numpy array

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> im_with_text = im_write_text(im, text = "hello world", x = 10, y = 10)
		>>> simple_im_diff(im, im_with_text)
		True
	'''
	pil_im = get_pil_im(im).copy()

	# font = ImageFont.truetype(font_path, size = font_size, encoding = 'unic')
	font = ImageFont.load_default()
	draw = ImageDraw.Draw(pil_im)
	w, h = draw.textsize(text, font)
	if isinstance(tup_bg_rgb, tuple):
		draw.rectangle(xy =(x,y,x+w,y+h), fill = tup_bg_rgb, outline = tup_bg_rgb)
	draw.text((x,y), text, tup_font_rgb, font = font)
	return np.array(pil_im)

def im_draw_bbox(im: Any, x0: int, y0: int, x1: int, y1:int,
		color: str = 'black', width:int = 3, caption: str = None,
		use_bbv: bool = False, bbv_label_only: bool = False
	) -> np.ndarray:
	''' returns a numpy image of an image with bounding box drawn on top

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		x0: bounding box top-left x location
		y0: bounding box top-left y location
		x1: bounding box bottom-right x location
		y1: bounding box bottom-right y location
		color: [HTML color name](https://www.w3schools.com/colors/colors_names.asp) as supported by [PIL's ImageColor](https://pillow.readthedocs.io/en/stable/reference/ImageColor.html)
		width: width of the bounding box lines
		caption: label for the bounding box
		use_bbv: use the package [`bbox_visualizer`](https://github.com/shoumikchow/bbox-visualizer) instead of PIL
		bbv_label_only: draw flag with label on the bounding box's centroid only

	Returns:
		a numpy array

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> bbox = {'x0': 10, 'y0': 10, 'x1': 190, 'y1': 290}
		>>> im_with_bbox = im_draw_bbox(im, caption = "hello world", **bbox)
		>>> simple_im_diff(im, im_with_bbox)
		True
	'''
	x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
	if use_bbv:
		import bbox_visualizer as bbv
		im = get_im(im)
		if bbv_label_only:
			if caption:
				im_array = bbv.draw_flag_with_label(im,
								label = caption,
								bbox = [x0,y0,x1,y1],
								line_color = ImageColor.getrgb(color),
								text_bg_color = ImageColor.getrgb(color)
							)
			else:
				raise ValueError(f'im_draw_bbox: bbv_label_only is True but caption is None')
		else:
			im_array = bbv.draw_rectangle(im,
							bbox = [x0, y0, x1, y1],
							bbox_color = ImageColor.getrgb(color),
							thickness = int(width)
						)
			im_array = bbv.add_label(im_array, label = caption,
							bbox = [x0,y0,x1,y1],
							text_bg_color = ImageColor.getrgb(color)
						) if caption else im_array
	else:
		pil_im = get_pil_im(im).copy()
		draw = ImageDraw.Draw(pil_im)
		draw.rectangle([(x0, y0), (x1, y1)], outline = color, width = int(width))
		if caption:
			pil_im = im_write_text(pil_im, text = caption, x = x0, y = y0,
						tup_font_rgb= ImageColor.getrgb(color))
		im_array = np.array(pil_im)
	return im_array

def im_centre_pad(im: Any, target_wh: Tuple[int,int] ) -> np.ndarray:
	'''resize an image to target_wh while keeping aspect ratio by [padding the borders to keep it centered](https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/)

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		target_wh: target width height in a Tuple

	Returns:
		a numpy array

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> im_centre_pad(im, target_wh = (300,300)).shape
		(300, 300, 3)
	'''
	p_im = get_pil_im(im)
	org_wh = p_im.size
	ratio = max(target_wh)/ max(org_wh)
	new_wh = tuple([int(x * ratio) for x in org_wh])
	p_im = p_im.resize(new_wh, Image.LANCZOS)
	out_im =  Image.new('RGB', target_wh)

	w,h = target_wh
	w_, h_ = new_wh
	paste_xy = ((w-w_)//2, (h-h_)//2)
	out_im.paste(p_im, paste_xy)
	return np.array(out_im)

def image_stack(im1: Any, im2: Any, do_vstack: bool = False, split_pct: float = None,
				black_line_thickness: int = 5,
				resample: int = Image.LANCZOS, debug: bool = False
	) -> np.ndarray:
	''' Concat two images into one with a border separating the two

	Args:
		im1: the left or top image
		im2: the right or bottom image (will be resize to match im1's dimension); **Aspect ratio** might change!
		do_vstack: do vertical stack, else, horizontal stack
		split_pct: Percentage of left image to show, right image will fill up the remainder. If none, both left and right images will be shown in full
		black_line_thickness: how many pixels wide is the black line separating the two images
		resample: one of [PIL Image resampling methods](https://note.nkmk.me/en/python-pillow-image-resize/)
		debug: print out which combine algo will be used on the two resized images

	Returns:
		a numpy array

	Raises:
		AssertionError: An error will be raise split_pct is greater than or equal to one

	Examples:
		>>> im1 = get_im("https://picsum.photos/200/300")
		>>> im2 = get_pil_im("https://picsum.photos/200/500")
		>>> image_stack(im1, im2, do_vstack = True, black_line_thickness = 5).shape
		(805, 200, 3)
	'''
	# input validation
	if split_pct:
		assert split_pct < 1, f"split_pct must be float and less than or equal to 1."

	img_arr_1, img_arr_2 = get_im(im1), get_im(im2)
	h, w, c = img_arr_1.shape
	_h, _w, _c = img_arr_2.shape
	if do_vstack:
		_h *= w/ _w
		_w = w
		np_black_line = np.zeros(shape = [black_line_thickness, w, 3], dtype = np.uint8)
	else:
		_w *= h/ _h
		_h = h
		np_black_line = np.zeros(shape = [h, black_line_thickness, 3], dtype = np.uint8)

	# image resize operation
	a_end = split_pct if split_pct else 1
	b_start = split_pct if split_pct else 0

	b_img = Image.fromarray(img_arr_2).resize(size = (int(_w),int(_h)), resample = resample)
	if do_vstack:
		a_img = img_arr_1[:int(h * a_end ), :, :]
		b_img = np.array(b_img)[int(_h * b_start):,:,:]
	else:
		a_img = img_arr_1[:, :int(w * a_end ), :]
		b_img = np.array(b_img)[:,int(w * b_start):,:]

	# alpha channel
	if c != _c:
		warnings.warn(f'image_stack: img_arr_1 channel {c} and img_arr_2 {_c} does not match. adding alpha to both.')
		a_img = im_add_alpha(a_img)
		b_img = im_add_alpha(b_img)
		np_black_line = im_add_alpha(np_black_line)

	np_func = np.vstack if do_vstack else np.hstack
	img_comb = np_func([a_img, np_black_line, b_img])
	if debug:
		print(f'applied {np_func} on im1 (shape: {a_img.shape}) and im2 (shape: {b_img.shape})')
	return img_comb

def im_mask_bbox(im: Any, l_bboxes: List[dict],
		mask_rgb_tup: Tuple[int, int, int] = (0,0,0),
		b_mask_bg: bool = True
	) -> np.ndarray:
	''' Mask the input image with the provided bounding boxes

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		l_bboxes: List of bounding boxes
		mask_rgb_tup: tuple of mask rgb value
		b_mask_bg: if True area outside the bounding boxes will be set to mask_rgb; otherwise, the area inside the bounding boxes will be set to mask_rgb

	Returns:
		a numpy array

	Examples:
		>>> im_org = get_im("https://picsum.photos/200/300")
		>>> bbox = {'x0': 10, 'y0': 10, 'x1': 190, 'y1': 290}
		>>> im_masked = im_mask_bbox(im_org, l_bboxes= [bbox])
		>>> simple_im_diff(im_org, im_masked)
		True
		>>> im_org.shape == im_masked.shape
		True
	'''
	im_rgb_array = get_im(im)
	h,w, c = im_rgb_array.shape
	bg_im = np.zeros([h,w,3], dtype = np.uint8) # black
	bg_im[:,:] = mask_rgb_tup # apply color
	im = np.copy(im_rgb_array)

	for bbox in l_bboxes:
		bg_im[bbox['y0']: bbox['y1'], bbox['x0']: bbox['x1']] = im_crop(im_rgb_array, **bbox)
		im[bbox['y0']: bbox['y1'], bbox['x0']: bbox['x1']] = mask_rgb_tup
	return bg_im if b_mask_bg else im

def im_color_mask(im: Any, mask_array: np.ndarray,
		rgb_tup: Tuple[int,int,int] = (91,86,188), alpha: float = 0.5
	) -> np.ndarray:
	''' draw mask over the input image

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		mask_array: a numpy array that should be of the same width-height as `im`
		rgb_tup: color of the mask in RGB
		alpha: level of transparency 0 being totally transparent, 1 solid color

	Returns:
		a numpy array in the shape of (h, w, 3)

	Raises:
		AssertionError: An error will be raise if image's width-height is not the same as the mask's width-height

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> mask = np.ones(im.shape[:2], dtype = np.uint8)
		>>> im_masked = im_color_mask(im, mask_array = mask)
		>>> simple_im_diff(im, im_masked)
		True
		>>> im_masked.shape == im.shape
		True
	'''
	im_rgb_array = get_im(im)
	assert im_rgb_array.shape[:2] == mask_array.shape[:2], \
		f'image is shape {im_rgb_array.shape[:2]} which is different than mask shape {mask_array.shape[:2]}'

	bg_im = np.zeros(im_rgb_array.shape, dtype = np.uint8) # create color
	bg_im[:,:]= rgb_tup
	im = Image.composite( Image.fromarray(bg_im), Image.fromarray(im_rgb_array),
						Image.fromarray(mask_array * int(alpha * 255))
						)
	return np.array(im)

def im_apply_mask(im : Any, mask_array: np.ndarray, mask_gblur_radius: int = 0,
		bg_rgb_tup: Tuple[int,int,int] = None, bg_blur_radius: int = None,
		bg_greyscale: bool = False,
	) -> np.ndarray:
	''' our more advanced image operation with mask [implemented using PIL](https://stackoverflow.com/questions/47723154/how-to-use-pil-paste-with-mask)

	With a mask, this function can create four different effects:
	background blurring, background removal (replace with a solid color),
	greyscaling background, and creating a 4-channel image.
	Additionally, a blur can be applied to the given mask to
	soften the edges. For details on picking a sensible blur radius
	[see here](https://stackoverflow.com/questions/62968174/for-pil-imagefilter-gaussianblur-how-what-kernel-is-used-and-does-the-radius-par).
	Note that **only ONE** of `[bg_rgb_tup, bg_blur_radius, bg_greyscale]`
	should be provided.

	Args:
		im: an image representation in numpy array, PIL Image object, URL, or filepath
		mask_array: a numpy array that should be of the same width-height as `im`
		mask_gblur_radius: mask's gaussian blur radius to soften the edges of the mask. [Implemented using PIL](https://stackoverflow.com/questions/62273005/compositing-images-by-blurred-mask-in-numpy), defaults to 0 (i.e. no blur applied).
		bg_rgb_tup: if given, return a 3-channel image with color background instead of transparent
		bg_blur_radius: if given, return a 3-channel image with GaussianBlur applied to the background
		bg_greyscale: color of the background (part outside the mask) in RGB

	Returns:
		a numpy array in the shape of (h, w, 3) if `bg_rgb_tup` or `bg_blur_radius` or `bg_greyscale` is provided; otherwise a numpy array of shape (h, w, 4) will be returned.

	Raises:
		AssertionError: An error will be raise if image's width-height is not the same as the mask's width-height, or if more than one of `[bg_rgb_tup, bg_blur_radius, bg_greyscale]` are provided.

	Examples:
		>>> im = get_im("https://picsum.photos/200/300")
		>>> mask = np.zeros(im.shape[:2], dtype = np.uint8)
		>>> mask[100:200, 50:150] = 1
		>>> im_masked_green = im_apply_mask(im, mask_array = mask, bg_rgb_tup = (50,205,50))
		>>> im_masked_gs = im_apply_mask(im, mask_array = mask, bg_greyscale = True)
		>>> im_masked_blur = im_apply_mask(im, mask_array = mask, bg_blur_radius = 5, mask_gblur_radius = 2)
		>>> im_masked_vanilla = im_apply_mask(im, mask_array = mask)
		>>> simple_im_diff(im, im_masked_green)
		True
		>>> simple_im_diff(im_masked_green, im_masked_gs)
		True
		>>> simple_im_diff(im_masked_gs, im_masked_blur)
		True
		>>> im_masked_gs.shape != im_masked_vanilla.shape
		True
		>>> im_masked_vanilla.shape
		(300, 200, 4)
	'''
	im_rgb_array = get_im(im)
	p_im = Image.fromarray(im_rgb_array)
	h, w, c = im_rgb_array.shape
	assert (h,w) == mask_array.shape[:2], \
		f"mask_array height-width {mask_array.shape} must match im_rgb_array's {(h, w)}"
	assert not(all([bg_rgb_tup, bg_blur_radius, bg_greyscale])), \
		f"only one of bg_rgb_tup, bg_blur_radius, or bg_greyscale call be specified."

	# convert bitwise mask from np to pillow
	# ref: https://note.nkmk.me/en/python-pillow-paste/
	pil_mask = Image.fromarray(np.uint8(255* mask_array))
	pil_mask = pil_mask.filter(
					ImageFilter.GaussianBlur(radius = mask_gblur_radius)
				) if mask_gblur_radius > 0 else pil_mask

	if bg_rgb_tup:
		bg_im = np.zeros([h,w,3], dtype = np.uint8) # black
		bg_im[:,:] = bg_rgb_tup						# apply color

		bg_im = Image.fromarray(bg_im)
		bg_im.paste(p_im, mask = pil_mask)
		p_im = bg_im
	elif bg_blur_radius:
		bg_im = p_im.copy().filter(
					ImageFilter.GaussianBlur(radius = bg_blur_radius)
				)
		bg_im.paste(p_im, mask = pil_mask)
		p_im = bg_im
	elif bg_greyscale:
		bg_im = ImageOps.grayscale(p_im)
		bg_im = np.array(bg_im)
		bg_im = np.stack((bg_im,)*3, axis = -1) 	# greyscale 1-channel to 3-channel

		bg_im =  Image.fromarray(bg_im)
		bg_im.paste(p_im, mask = pil_mask)
		p_im = bg_im
	else:
		p_im.putalpha(pil_mask)

	return np.array(p_im)
#
# def mask_overlap(base_mask, over_mask, get_overlap_mask = False):
# 	'''
# 	compute the percentage of mask union
# 	Args:
# 		get_overlap_mask: if true it will return a mask of only the union
# 	'''
# 	if base_mask.shape != over_mask.shape:
# 		raise ValueError(f'mask_overlap: base_mask shape {base_mask.shape} does not match over_mask {over_mask.shape}')
#
# 	overlap = np.logical_and(base_mask!= 0, over_mask != 0)
# 	score = (overlap== True).sum() / np.count_nonzero(base_mask)
# 	return overlap.astype(np.uint8) if get_overlap_mask else float(score)
#
# def join_binary_masks(list_of_np_binary_masks):
# 	l_masks = list_of_np_binary_masks
# 	for mk in l_masks:
# 		if mk.shape != l_masks[0].shape:
# 			raise ValueError(f'join_binary_masks: all masks must be of the same shape')
#
# 	out_mask = l_masks[0]
# 	for mk in l_masks[1:]:
# 		out_mask += mk
# 	return np.array(out_mask!=0).astype(np.uint8)
#
# def mask_bbox(input_mask, get_json = False):
# 	'''
# 	get the minimum bounding box of a np binary mask
# 	returns y0,y1,x0,x1
# 	'''
# 	rows = np.any(input_mask, axis=1) # y-axis
# 	cols = np.any(input_mask, axis=0) # x-axis
# 	rmin, rmax = np.where(rows)[0][[0, -1]]
# 	cmin, cmax = np.where(cols)[0][[0, -1]]
# 	rmin, rmax, cmin, cmax = list(map(int,[rmin,rmax,cmin,cmax]))
# 	return {'x0': cmin, 'x1': cmax, 'y0': rmin, 'y1': rmax} if get_json else (rmin, rmax, cmin, cmax)
#
#
#
# def plot_colors(hist, centroids, w= 300 , h = 50):
# 	'''
# 	return a pil_im of color given in centroids
# 	'''
# 	# initialize the bar chart representing the relative frequency of each of the colors
# 	bar = np.zeros((50, 300, 3), dtype = "uint8")
# 	startX = 0
#
# 	im = Image.new('RGB', (300, 50), (128, 128, 128))
# 	draw = ImageDraw.Draw(im)
#
# 	# loop over the percentage of each cluster and the color of
# 	# each cluster
# 	for (percent, color) in zip(hist, centroids):
# 		# plot the relative percentage of each cluster
# 		endX = startX + (percent * 300)
# 		xy = (int(startX), 0, int(endX), 50)
# 		fill = tuple(color.astype('uint8').tolist())
# 		draw.rectangle(xy, fill)
# 		startX = endX
#
# 	# return the bar chart
# 	im.resize( (w,h))
# 	return im
#
# def gamma_adjust(pil_im, gamma = 1):
# 	'''
# 	return a PIL Image with gamma correction (brightness)
# 	see: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# 	and code: https://note.nkmk.me/en/python-numpy-image-processing/
# 	'''
# 	im = np.array(pil_im)
# 	im_out = 255.0 * (im/ 255.0)**(1/gamma)
# 	return Image.fromarray(im_out.astype(np.uint8))
