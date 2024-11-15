# 대본
# [Slide 01]
# Hello everyone! My name is Hanbyeol Lee from Barami’s 27th class, and I’m excited to lead today’s seminar on 3D printing.

# Over the next six weeks, we’ll embark on a journey into the world of 3D printing. From weeks two to four, we’ll dive into theoretical lectures to build a solid foundation. Then, in weeks five and six, we’ll get hands-on with project sessions. You’ll create your own work plans and experience 3D printing firsthand. Finally, in the sixth week, we’ll come together to present and share our projects.

# [Slide 02]
# Let’s begin by exploring the three main types of 3D modeling methods: NURBS, Solid, and Polygon modeling. NURBS, which stands for Non-Uniform Rational B-Splines, uses mathematical functions to create smooth and complex curved surfaces. This method offers precise control over intricate designs but requires a good understanding of mathematical formulas, which can be a bit challenging.

# Next is Polygon Modeling, where the basic unit is the triangle. By connecting vertices, we build our 3D models. The level of detail depends on the number of polygons used. High-resolution models use a large number of polygons, resulting in detailed models but slower rendering speeds. Low-resolution models use fewer polygons, leading to faster rendering but less detail. The key difference lies in balancing detail and performance based on the polygon count.

# Then we have Solid Modeling, which involves creating models by matching surfaces together to form a fully filled object. These models are solid inside, unlike hollow ones. While they can be heavier and more resource-intensive, they allow us to calculate physical properties like weight and volume accurately.

# [Slide 03]
# Now, let’s take a look at the 3D workspace, also known as the viewport. The primary view is the Perspective View, which provides an overall look at your model in a realistic, three-dimensional space. In addition to the perspective view, there are three essential orthographic views we need to consider: Top View, Front View, and Left View.

# The Top View shows the model from above and is useful for understanding the layout and positioning of different parts. The Front View displays the model from the front, helping us analyze height and vertical details. The Left View provides a side perspective from the left, which is useful for examining depth and side features. When you’re drafting your work plans, it’s crucial to include drawings from all three of these views to ensure a comprehensive understanding of your design from every angle.

# [Slide 04]
# Moving on to how to write a work instruction sheet, this document guides the entire modeling and printing process. The key components are design requirements, information extraction, and drawing the blueprint. In the design requirements, you outline what you aim to achieve with your design, specifying dimensions and any constraints—precise measurements are vital here. Information extraction involves gathering all necessary data and specifications needed for your model, such as materials, tools, or reference models. Drawing the blueprint means creating detailed sketches of your model from different views, including all measurements and dimensions.

# For instance, if you’re designing a donut-shaped object with an outer diameter of 5 units and an inner diameter of 3 units, the thickness would be 2 units. Including such precise details in your blueprint is essential for accurate modeling.

# [Slide 05]
# Now, let’s discuss how to prepare your data for printing by adjusting the 3D printer settings. Several key factors need consideration, such as resolution settings, printing speed, temperature settings, supports, and infill density.

# Resolution settings determine the level of detail and printing time. A 0.4 mm resolution produces rougher models with lower detail but faster printing times. A 0.2 mm resolution offers a good balance between detail and print speed and is commonly used. A 0.1 mm resolution yields highly detailed models but significantly increases printing times.

# Printing speed is generally preset, so there’s usually no need to adjust it unless you have specific requirements. For temperature settings, set the nozzle temperature to around 210°C and the bed temperature to about 50°C when using PLA filament. Supports are necessary for models with overhangs or intricate features to prevent sagging during the printing process. Infill density determines how solid the inside of your model is and is commonly set between 10% to 20%, balancing structural integrity with material usage.

# We’ll be using the slicing software CURA during our seminar. It’s a widely used and versatile program that’s great for beginners and experts alike. For materials, we’ll work with PLA filament made from corn starch—it’s safe, easy to use, and environmentally friendly.

# That brings us to the end of today’s seminar. Thank you all for your attention and participation. I’m really looking forward to seeing the amazing projects you’ll create in the upcoming weeks! If you have any questions or need assistance, please don’t hesitate to reach out. Let’s make the most of this learning experience together.

# Thank you!”


from transformers import pipeline

# Use a pipeline as a high-level helper
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# 사용 예시
text = '''
Now, let’s discuss how to prepare your data for printing by adjusting the 3D printer settings. Several key factors need consideration, such as resolution settings, printing speed, temperature settings, supports, and infill density.

Resolution settings determine the level of detail and printing time. A 0.4 mm resolution produces rougher models with lower detail but faster printing times. A 0.2 mm resolution offers a good balance between detail and print speed and is commonly used. A 0.1 mm resolution yields highly detailed models but significantly increases printing times.

Printing speed is generally preset, so there’s usually no need to adjust it unless you have specific requirements. For temperature settings, set the nozzle temperature to around 210°C and the bed temperature to about 50°C when using PLA filament. Supports are necessary for models with overhangs or intricate features to prevent sagging during the printing process. Infill density determines how solid the inside of your model is and is commonly set between 10% to 20%, balancing structural integrity with material usage.

We’ll be using the slicing software CURA during our seminar. It’s a widely used and versatile program that’s great for beginners and experts alike. For materials, we’ll work with PLA filament made from corn starch—it’s safe, easy to use, and environmentally friendly.

That brings us to the end of today’s seminar. Thank you all for your attention and participation. I’m really looking forward to seeing the amazing projects you’ll create in the upcoming weeks! If you have any questions or need assistance, please don’t hesitate to reach out. Let’s make the most of this learning experience together.

Thank you!
'''
summary = summarizer(text, max_length=100, min_length=40, do_sample=False)
print(summary[0]["summary_text"])