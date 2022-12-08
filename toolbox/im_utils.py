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
from typing import Union

def get_pil_im(fp_url_nparray: Union[str, np.ndarray]) -> Union[Image.Image, None]:
	''' return a PIL image object

	Args:
		fp_url_nparray: filepath, url, or np.array

	Returns:
		A PIL Image object

	Raises:
		ValueError: An error when fp_url_nparray cannot be converted into an image

	Examples:
		>>> get_pil_im("https://random.imagecdn.app/500/150").size
		(500, 150)
	'''
	import urllib.request as urllib
	from urllib.error import HTTPError

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
			pil_im = None
	else:
		raise ValueError(f'get_im: im must be np array, filename, or url')
	return pil_im

def im_crop(im_rgb_array, x0, y0, x1, y1):
	return im_rgb_array[y0: y1, x0:x1, :]

def im_mask_bbox(im_rgb_array, l_bboxes, mask_rgb_tup = (0,0,0), bMaskBG = True):
    ''' Mask the input image with the provided bounding boxes
    Args:
        bMaskBG: if True area outside the bounding boxes will be set to mask_rgb
            otherwise, the area inside the bounding boxes will be set to mask_rgb
    '''
    h,w, c = im_rgb_array.shape
    bg_im = np.zeros([h,w,3], dtype = np.uint8) # black
    bg_im[:,:] = mask_rgb_tup # apply color
    im = np.copy(im_rgb_array)

    for bbox in l_bboxes:
        bg_im[bbox['y0']: bbox['y1'], bbox['x0']: bbox['x1']] = im_crop(im_rgb_array, **bbox)
        im[bbox['y0']: bbox['y1'], bbox['x0']: bbox['x1']] = mask_rgb_tup
    return bg_im if bMaskBG else im

def im_color_mask(im_rgb_array, mask_array, rbg_tup = (91,86,188), alpha = 0.5, get_pil_im = False):
	'''
	return np image with mask drawn over input image
	Args:
		alpha: level of transparency 0 being totally transparent, 1 solid color
	'''
	if im_rgb_array.shape[:1] != mask_array.shape[:1]:
		raise ValueError(f'im_color_mask: image is shape {im_rgb_array.shape[:2]} which is different than mask shape {mask_array.shape[:2]}')

	bg_im = np.zeros(im_rgb_array.shape, dtype = np.uint8) # create color
	bg_im[:,:]= rbg_tup
	im = Image.composite( Image.fromarray(bg_im), Image.fromarray(im_rgb_array),
						Image.fromarray(mask_array * int(alpha * 255))
						)
	return im if get_pil_im else np.array(im)

def im_apply_mask(im_rgb_array, mask_array, get_pil_im = False, bg_rgb_tup = None,
	bg_blur_radius = None, bg_greyscale = False, mask_gblur_radius = 0):
	'''
	return either a np array with 4 channels or PIL Image with alpha
	ref: https://stackoverflow.com/questions/47723154/how-to-use-pil-paste-with-mask
	ref: https://stackoverflow.com/questions/62273005/compositing-images-by-blurred-mask-in-numpy
	ref: https://stackoverflow.com/questions/62968174/for-pil-imagefilter-gaussianblur-how-what-kernel-is-used-and-does-the-radius-par

	Args:
		bg_rgb_tup: if given, return a 3-channel image with color background instead of transparent
		bg_blur_radius: if given, return a 3-channel image with GaussianBlur applied to the background
	'''
	h, w, c = im_rgb_array.shape
	m_h, m_w = mask_array.shape

	if not all([h == m_h, w == m_w]):
		raise ValueError(f'im_apply_mask: mask_array size {(m_h, m_w)} must match im_rgb_array {(h, w)}')

	im = Image.fromarray(im_rgb_array)

	# convert bitwise mask from np to pillow
	# ref: https://note.nkmk.me/en/python-pillow-paste/
	pil_mask = Image.fromarray(np.uint8(255* mask_array))
	pil_mask = pil_mask.filter(
					ImageFilter.GaussianBlur(radius = mask_gblur_radius)
				) if mask_gblur_radius > 0 else pil_mask

	if bg_rgb_tup:
		bg_im = np.zeros([h,w,3], dtype = np.uint8) # black
		bg_im[:,:] = bg_rgb_tup						# apply color

		# old method using just np but doesn't support blurred mask
		# idx = (mask_array != 0)
		# bg_im[idx] = im_rgb_array[idx]

		bg_im = Image.fromarray(bg_im)
		bg_im.paste(im, mask = pil_mask)
		im = bg_im
	elif bg_blur_radius:
		bg_im = im.copy().filter(
					ImageFilter.GaussianBlur(radius = bg_blur_radius)
				)
		bg_im.paste(im, mask = pil_mask)
		im = bg_im
	elif bg_greyscale:
		bg_im = ImageOps.grayscale(Image.fromarray(im_rgb_array))
		bg_im = np.array(bg_im)
		bg_im = np.stack((bg_im,)*3, axis = -1) 	# greyscale 1-channel to 3-channel

		bg_im =  Image.fromarray(bg_im)
		bg_im.paste(im, mask = pil_mask)
		im = bg_im
	else:
		im.putalpha(pil_mask)

	return im if get_pil_im else np.array(im)

def mask_overlap(base_mask, over_mask, get_overlap_mask = False):
	'''
	compute the percentage of mask union
	Args:
		get_overlap_mask: if true it will return a mask of only the union
	'''
	if base_mask.shape != over_mask.shape:
		raise ValueError(f'mask_overlap: base_mask shape {base_mask.shape} does not match over_mask {over_mask.shape}')

	overlap = np.logical_and(base_mask!= 0, over_mask != 0)
	score = (overlap== True).sum() / np.count_nonzero(base_mask)
	return overlap.astype(np.uint8) if get_overlap_mask else float(score)

def join_binary_masks(list_of_np_binary_masks):
	l_masks = list_of_np_binary_masks
	for mk in l_masks:
		if mk.shape != l_masks[0].shape:
			raise ValueError(f'join_binary_masks: all masks must be of the same shape')

	out_mask = l_masks[0]
	for mk in l_masks[1:]:
		out_mask += mk
	return np.array(out_mask!=0).astype(np.uint8)

def mask_bbox(input_mask, get_json = False):
	'''
	get the minimum bounding box of a np binary mask
	returns y0,y1,x0,x1
	'''
	rows = np.any(input_mask, axis=1) # y-axis
	cols = np.any(input_mask, axis=0) # x-axis
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]
	rmin, rmax, cmin, cmax = list(map(int,[rmin,rmax,cmin,cmax]))
	return {'x0': cmin, 'x1': cmax, 'y0': rmin, 'y1': rmax} if get_json else (rmin, rmax, cmin, cmax)

def im_draw_bbox(pil_im, x0, y0, x1, y1, color = 'black', width = 3, caption = None,
					caption_font = ImageFont.load_default(),
					use_bbv = False, bbv_label_only = False):
	'''
	draw bounding box on the input image pil_im in-place
	Args:
		color: color name as read by Pillow.ImageColor
		use_bbv: use bbox_visualizer
	'''
	if any([type(i)== float for i in [x0,y0,x1,y1]]):
		warnings.warn(f'im_draw_bbox: at least one of x0,y0,x1,y1 is of the type float and is converted to int.')
		x0 = int(x0)
		y0 = int(y0)
		x1 = int(x1)
		y1 = int(y1)

	if use_bbv:
		import bbox_visualizer as bbv
		if bbv_label_only:
			if caption:
				im_array = bbv.draw_flag_with_label(np.array(pil_im),
							label = caption,
							bbox = [x0,y0,x1,y1],
							line_color = ImageColor.getrgb(color),
							text_bg_color = ImageColor.getrgb(color)
							)
			else:
				raise ValueError(f'im_draw_bbox: bbv_label_only is True but caption is None')
		else:
			im_array = bbv.draw_rectangle(np.array(pil_im),
						bbox = [x0, y0, x1, y1],
						bbox_color = ImageColor.getrgb(color),
						thickness = int(width)
						)
			im_array = bbv.add_label(
						im_array, label = caption,
						bbox = [x0,y0,x1,y1],
						text_bg_color = ImageColor.getrgb(color)
						)if caption else im_array
		return Image.fromarray(im_array)
	else:
		draw = ImageDraw.Draw(pil_im)
		draw.rectangle([(x0, y0), (x1, y1)], outline = color, width = int(width))
		if caption:
			draw.text((x0, y0), text = caption, fill = color, font = caption_font)
	#return None

def downsize_im(pil_im, max_h = 100):
	o_w, o_h = pil_im.size
	h = min(o_h, max_h)
	w = int(o_w * h/ o_h)
	im_small = pil_im.resize((w,h), Image.ANTIALIAS) #best downsize filter
	return im_small

def im_max_short_edge(im_np_array, size, return_pil_im = False,
		resample_algo = Image.LANCZOS, debug = False):
	''' Return an image whose short edge is no longer than the given size
	Args:
		resample_algo: default to LANCZOS b/c it gives best downscaling quality (per https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table)
	'''
	org_h, org_w, _ = im_np_array.shape
	out_im = None
	if debug:
		print(f'im_max_short_edge: seeing input w,h of {(org_w, org_h)}')

	if min(org_h, org_w) <= size:
		out_im = im_np_array
		if debug:
			print(f'im_max_short_edge: image dim is smaller than max {size}. no resizing required.')
	else:
		wh_ratio = org_w / org_h
		if org_h > org_w:
			# fix w to size
			w = size
			h = w / wh_ratio
		else:
			# fix h to size
			h = size
			w = h * wh_ratio
		w = int(w)
		h = int(h)
		pil_im = Image.fromarray(im_np_array).resize((w,h), resample = resample_algo)
		out_im = np.array(pil_im)

		if debug:
			print(f'im_max_short_edge: resizing image to w,h of {(w,h)}')
	return Image.fromarray(out_im) if return_pil_im else out_im

def im_min_short_edge(im_np_array, size, return_pil_im = False,
		resample_algo = Image.LANCZOS, debug = False):
	''' Return an image whose short edge is no shorter than the given size
	Args:
		resample_algo: default to LANCZOS b/c it gives best downscaling quality (per https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table)
	'''
	org_h, org_w, _ = im_np_array.shape
	out_im = None
	if debug:
		print(f'im_min_short_edge: seeing input w,h of {(org_w, org_h)}')

	if min(org_h, org_w) >= size:
		out_im = im_np_array
		if debug:
			print(f'im_min_short_edge: image dim is already larger than max {size}. no resizing required.')
	else:
		wh_ratio = org_w / org_h
		if org_h > org_w:
			# fix w to size
			w = size
			h = w / wh_ratio
		else:
			# fix h to size
			h = size
			w = h * wh_ratio
		w = int(w)
		h = int(h)
		pil_im = Image.fromarray(im_np_array).resize((w,h), resample = resample_algo)
		out_im = np.array(pil_im)

		if debug:
			print(f'im_min_short_edge: resizing image to w,h of {(w,h)}')
	return Image.fromarray(out_im) if return_pil_im else out_im

def image_stack(img_arr_1, img_arr_2, do_vstack = False, split_pct = None, black_line_thickness = 5,
				resample = Image.LANCZOS, debug = False):
	'''
	return one image of left_img_arr and right_img_arr join together
	Args:
		img_arr_1: the left or top image
		img_arr_2: the right or bottom image (will be resize to match img_arr_1)
		do_vstack: do vertical stack, else, horizontal stack
		split_pct: Percentage of left image to show, right image will fill up the remainder. If none, both left and right images will be shown in full
		resample: one of PIL Image resampling methods
	'''
	# input validation
	if split_pct:
		if split_pct > 1:
			raise TypeError(f"split_pct must be float and less than or equal to 1.")

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
	if debug:
		print(f'apply {np_func} on img_arr_1 {a_img.shape} and img_arr_2 {b_img.shape}')
	img_comb = np_func([a_img, np_black_line, b_img])

	return img_comb

def im_add_alpha(img_arr):
	'''
	add an empty alpha channel if img_arr only has 3 channels
	'''
	h,w,c = img_arr.shape
	if c == 3:
		return np.concatenate((
				img_arr, np.ones((h, w, 1))+254
				), axis =2).astype('uint8')
	else:
		warnings.warn(f'im_add_alpha: input image array does not have 3 channels. Returning input array.')
		return img_arr

def im_gray2rgb(pil_im):
	return pil_im.copy().convert('RGB')

def im_3c(img_arr):
	'''return only 3 channel for RGB'''
	assert isinstance(img_arr, np.ndarray), f'img_arr needs to be a numpy array'
	assert len(img_arr.shape) == 3, f'img_arr expected to a 3D array but has shape of {img_arr.shape}'
	if img_arr.shape[2] < 3: #, f'expecting at least 3 channels'
		img_arr = np.array(im_gray2rgb(Image.fromarray(img_arr)))
	return img_arr[:,:,:3]

def image_to_bytes_array(PIL_Image, format = None, quality = 100):
	'''
	Takes a PIL Image and convert it to Binary Bytes
	PIL_Image.tobytes() is not recommended (see: https://stackoverflow.com/a/58949303/14285096)

	Args:
		quality: compression quality. passed to Image.save() (see: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg)
	'''
	imgByteArr = io.BytesIO()
	# from the PIL doc:
	# If a file object was used instead of a filename, this parameter should always be used.
	img_format = format if format else PIL_Image.format
	img_format = img_format if img_format else 'jpeg'
	PIL_Image.save(imgByteArr, format = img_format, quality = quality)
	imgByteArr = imgByteArr.getvalue()
	return imgByteArr

def plot_colors(hist, centroids, w= 300 , h = 50):
	'''
	return a pil_im of color given in centroids
	'''
	# initialize the bar chart representing the relative frequency of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	im = Image.new('RGB', (300, 50), (128, 128, 128))
	draw = ImageDraw.Draw(im)

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		xy = (int(startX), 0, int(endX), 50)
		fill = tuple(color.astype('uint8').tolist())
		draw.rectangle(xy, fill)
		startX = endX

	# return the bar chart
	im.resize( (w,h))
	return im

def gamma_adjust(pil_im, gamma = 1):
	'''
	return a PIL Image with gamma correction (brightness)
	see: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	and code: https://note.nkmk.me/en/python-numpy-image-processing/
	'''
	im = np.array(pil_im)
	im_out = 255.0 * (im/ 255.0)**(1/gamma)
	return Image.fromarray(im_out.astype(np.uint8))
