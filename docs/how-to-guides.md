This part of the project documentation focuses on a
**problem-oriented** approach. You'll tackle common
tasks that you might have, with the help of the code
provided in this project.

## How to work with Images in Python

You have images and you want to do something with them.
You're in luck! The `toolbox` package can help you
get this done.

Install the `miro-vision-toolbox` with a `pip install miro-vision-toolbox`

Inside of `your_script.py` you can now import the
`get_pil_im()` function from the `toolbox.img_utils`
module:

    # your_script.py
    from toolbox.img_utils import get_pil_im

After you've imported the function, you can use it
to load any images you need:

    # your_script.py
    from toolbox.img_utils import get_pil_im

    print(get_pil_im("https://random.imagecdn.app/500/150"))  # OUTPUT: <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=500x150 at 0x1106B3810>

You're now able to get a PIL image object, and you can do
more things with it!
