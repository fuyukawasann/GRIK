from Utils.makePDF import makePDF as mpdf


SSIM_path = 'Result/Test/SSIM'
extract_path = 'Result/Test/Extracted'
result_name = 'Test'
this_object = mpdf(SSIM_path, extract_path, result_name)
this_time = this_object.make_pdf()


print("this_time is", this_time)