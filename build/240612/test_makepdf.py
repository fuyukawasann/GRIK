from Utils.makePDF import makePDF as mpdf

result_name = 'Test_200_16_ver3_20240613182305'
SSIM_path = f'Result/{result_name}/SSIM'
extract_path = f'Result/{result_name}/Extracted'
this_object = mpdf(SSIM_path, extract_path, result_name)
this_time = this_object.make_pdf()


print("this_time is", this_time)
