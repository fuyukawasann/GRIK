##### Make PDF File #####
# This file is used to make PDF file with the extracted image and handwritten image
# The extracted image is inserted at the middle of the PDF file
# BUILD: Jun 12, 2024
#########################

import os
import time
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Image, Paragrah, Spacer, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    os.system('pip install reportlab')
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet
import natsort
from PIL import Image as PILImage
    
class makePDF:
    def __init__(self, SSIM_path, extract_path, result_name):
        self.SSIM_path = SSIM_path
        self.extract_path = extract_path # Extracted Image -> These are inserted at the middle of the PDF
        self.result_name = result_name # Result File Name
        
    
    def make_pdf(self):
        # Generate PDF Document
        self.height, self.width = A4
        self.file_name = f'{self.result_name}.pdf'
        doc = SimpleDocTemplate(self.file_name, pagesize=A4, topMargin=0*inch, bottomMargin=0*inch)
        styles = getSampleStyleSheet()
        elements = []

		# Add Original Image and sorted the list in order
        self.original_image_path = f'{self.SSIM_path}/original'
        self.original_image = os.listdir(self.original_image_path)
        self.original_image = natsort.natsorted(self.original_image)
        # Add Handwritten Image
        self.handwritten_image_path = f'{self.SSIM_path}/handwritten'
        self.handwritten_image = os.listdir(self.handwritten_image_path)
        self.handwritten_image = natsort.natsorted(self.handwritten_image)
        
        # Set Start Time
        start_time = time.time()
        
		# Make Page
        for iter in range(len(self.original_image)):
            # Above Image (Handwritten Image)
            hw_img = PILImage.open(f'{self.handwritten_image_path}/{self.handwritten_image[iter]}')
            hw_img_width, hw_img_height = hw_img.size
            if hw_img_width > self.width and hw_img_height > self.height:
                ratio = min(self.width/hw_img_width, self.height/hw_img_height)
                new_width = hw_img_width * ratio * 0.45
                new_height = hw_img_height * ratio * 0.45
                elements.append(Image(f'{self.handwritten_image_path}/{self.handwritten_image[iter]}', width=new_width, height=new_height))
            elif hw_img_width > self.width:
                ratio = self.width/hw_img_width
                new_width = hw_img_width * ratio * 0.45
                new_height = hw_img_height * ratio * 0.45
                elements.append(Image(f'{self.handwritten_image_path}/{self.handwritten_image[iter]}', width=new_width, height=new_height))
            elif hw_img_height > self.height:
                ratio = self.height/hw_img_height
                new_width = hw_img_width * ratio * 0.45
                new_height = hw_img_height * ratio * 0.45
                elements.append(Image(f'{self.handwritten_image_path}/{self.handwritten_image[iter]}', width=new_width, height=new_height))
            else: elements.append(Image(f'{self.handwritten_image_path}/{self.handwritten_image[iter]}', width=hw_img_width*0.45, height=hw_img_height*0.45))
            elements.append(Spacer(1, 0.2*inch))
			
			# Add Extracted Image
            extract_img = [f for f in os.listdir(self.extract_path) if f.startswith(f'Test_{iter}_detect')]
            ## Sorted
            extract_img = natsort.natsorted(extract_img)
            for ext_iter in range(len(extract_img)):
                here_img = PILImage.open(f'{self.extract_path}/{extract_img[ext_iter]}')
                img_width, img_height = here_img.size
                if img_width > self.width and img_height > self.height:
                    ratio = min(self.width/img_width, self.height/img_height)
                    new_width = img_width * ratio * 0.4
                    new_height = img_height * ratio * 0.4
                    elements.append(Image(f'{self.extract_path}/{extract_img[ext_iter]}', width=new_width, height=new_height))
                elif img_width > self.width:
                    ratio = self.width/img_width
                    new_width = img_width * ratio * 0.4
                    new_height = img_height * ratio * 0.4
                    elements.append(Image(f'{self.extract_path}/{extract_img[ext_iter]}', width=new_width, height=new_height))
                elif img_height > self.height:
                    ratio = self.height/img_height
                    new_width = img_width * ratio * 0.4
                    new_height = img_height * ratio * 0.4
                    elements.append(Image(f'{self.extract_path}/{extract_img[ext_iter]}', width=new_width, height=new_height))
                else:
                    elements.append(Image(f'{self.extract_path}/{extract_img[ext_iter]}', width=img_width * 0.4, height=img_height * 0.4))
                ### If last Image set spacer greater than before
                if ext_iter != len(extract_img)-1:
                    elements.append(Spacer(1, 0.1*inch))
                else :
                    elements.append(Spacer(1, 0.2*inch))
            
			# Below Image (Original Image)
            ori_img = PILImage.open(f'{self.original_image_path}/{self.original_image[iter]}')
            ori_img_width, ori_img_height = ori_img.size
            if ori_img_width > self.width and ori_img_height > self.height:
                ratio = min(self.width/ori_img_width, self.height/ori_img_height)
                new_width = ori_img_width * ratio * 0.45
                new_height = ori_img_height * ratio * 0.45
                elements.append(Image(f'{self.original_image_path}/{self.original_image[iter]}', width=new_width, height=new_height))
            elif ori_img_width > self.width:
                ratio = self.width/ori_img_width
                new_width = ori_img_width * ratio * 0.45
                new_height = ori_img_height * ratio * 0.45
                elements.append(Image(f'{self.original_image_path}/{self.original_image[iter]}', width=new_width, height=new_height))
            elif ori_img_height > self.height:
                ratio = self.height/ori_img_height
                new_width = ori_img_width * ratio * 0.45
                new_height = ori_img_height * ratio * 0.45
                elements.append(Image(f'{self.original_image_path}/{self.original_image[iter]}', width=new_width, height=new_height))
            else: elements.append(Image(f'{self.original_image_path}/{self.original_image[iter]}', width=ori_img_width*0.45, height=ori_img_height*0.45))
			
   			# Add Page Break -> Next Page
            if(iter != len(self.original_image)-1):
                elements.append(PageBreak())
        
        # Generate PDF
        doc.build(elements)
        ## End Time
        end_time = time.time()
        ## Alert End
        print(f'PDF File is generated: {self.result_name}.pdf')
        
        # Eval Time
        eval_time = end_time - start_time
        print(f'Evaluation Time: {eval_time} sec')
        time.sleep(1)
        
        
        # Move the PDF File to the Result Folder
        print("Move PDF File to the Result Folder...")
        os.system(f'mv {self.file_name} "Result/{self.result_name}/{self.file_name}"')
        print("Move PDF File to the Result Folder... Success")
        
        return eval_time