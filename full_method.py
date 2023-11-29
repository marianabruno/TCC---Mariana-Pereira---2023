from image_processing import *
from IPM import Plane, load_camera_params, bilinear_sampler, ipm_from_parameters
from statistics import median, pstdev

video = cv2.VideoCapture('straight_lane.mp4')
lines = Lines()
lines_bev = Lines()
vanish_point_accum = Accumulator(3)
TARGET_H, TARGET_W = 500, 500
angle = []
dist = []
pitch = []
yaw = []
# count=0

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('output.avi', fourcc, 30.35, (500,  500))

while video.isOpened():
    # Capture frame-by-frame
    ret, frame = video.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # count = count+1
    ###################################
    ###   PROCESSAMENTO DA IMAGEM   ###
    ###################################

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # transformação gamma para aumentar contraste entre asfalto e faixas
    gamma_frame = np.uint8((355/1000000) * (gray_frame ** 2.5))
    
    # segmentação da faixas por binarização; a binarização é feita por faixas na região de interesse
    # região de interesse: gamma_frame[0:360, :] (metade superior da imagem) 
    segmented_frame = np.zeros((720, 720), dtype=np.uint8)
    for i in range(10):
        slice_size = 36 
        a = i*slice_size
        b = (i+1)*slice_size   
        slice  = gamma_frame[a:b,:]
        threshold, binary_slice  = cv2.threshold(slice,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if threshold > 90:
            segmented_frame[a:b,:] = binary_slice
 
    skeleton_frame = skeletize_image(segmented_frame)

    ######################################
    ###   ESTIMATIVA DO VANISH POINT   ###
    ######################################

    lines.all_lines = hough_transform(skeleton_frame)
    lines.filter_by_angle(0.9)
    lines.sort_sides_by_angle()
    lines.buffer()

    normalize_hough(lines.left_lines)
    normalize_hough(lines.right_lines)

    avg_left_line, avg_right_line = lines.get_average_line()
    final_lines = np.concatenate((avg_left_line,avg_right_line), axis=0)
    final_lines_display = [avg_left_line, avg_right_line]

    v = vanish_point(final_lines)
    vanish_point_accum.accumulate(v)
    v_avg = vanish_point_accum.accumulator_average()
    v_avg = v_avg.astype(int)
    v_avg = np.reshape(v_avg,(-1))

    # ###########################
    # ###   BIRD'S EYE VIEW   ###
    # ###########################

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    interpolation_fn = bilinear_sampler
    # Define the plane on the region of interest (road)
    plane = Plane(0, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.1)
    # Retrieve camera parameters
    extrinsic, intrinsic, new_pitch, new_yaw = load_camera_params(v_avg)
    # pitch.append(new_pitch)
    # yaw.append(new_yaw)
    # Apply perspective transformation
    img_bev = ipm_from_parameters(image[:,:], plane.xyz, intrinsic, extrinsic, interpolation_fn, TARGET_H, TARGET_W)
    img_bev = cv2.rotate(img_bev, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # #############################
    # ###   ERROR CALCULATION   ###
    # #############################

    # gray_bev = cv2.cvtColor(img_bev, cv2.COLOR_BGR2GRAY)
    # gamma_bev = np.uint8((355/1000000) * (gray_bev ** 2.5))
    # threshold, binary_bev  = cv2.threshold(gamma_bev,90,255,cv2.THRESH_BINARY)
    # skeleton_bev = skeletize_image(binary_bev)

    # lines_bev.all_lines = hough_transform(skeleton_bev)
    # lines_bev.sort_sides_by_roi(230, 250, 250, 270)
    # lines_bev.buffer()

    # lines_bev.left_lines = normalize_hough(lines_bev.left_lines)
    # lines_bev.right_lines = normalize_hough(lines_bev.right_lines)

    # avg_left_line_bev, avg_right_line_bev = lines_bev.get_average_line()
    # final_lines_bev = np.concatenate((avg_left_line_bev,avg_right_line_bev), axis=0)
    # lines_bev_print = [avg_left_line_bev, avg_right_line_bev]

    # angle.append(abs(avg_left_line_bev[0,1])+abs(avg_right_line_bev[0,1]))
    # b=avg_right_line_bev[0,1]-avg_left_line_bev[0,1]
    # c=np.power(avg_right_line_bev[0,0],2)
    # d=np.power(avg_left_line_bev[0,0],2)
    # a = np.sqrt((c)+(d)-(2*avg_right_line_bev[0,0]*avg_left_line_bev[0,0]*np.cos(b)))
    # dist.append(a)
  
###################################################################################################################
    # display_lines(img_bev, lines_bev.left_lines, (0,255,0))
    # cv2.circle(skeleton_frame, (v_avg[0],v_avg[1]), 3, (0,255,255),5)
    # cv2.rectangle(frame, (0,60), (720,360), (255,0,0))
    
    # comparison = np.copy(frame)
    # comparison[:,:,2] = cv2.add(comparison[:,:,2], skeleton_frame)
    
    # cv2.imshow('comparison', img_bev) 
    # if cv2.waitKey(15) == ord('q'):
    #     break

    cv2.imshow('original', frame) 
    if cv2.waitKey(15) == ord('q'):
        break

    cv2.imshow('BEV', img_bev) 
    if cv2.waitKey(15) == ord('q'):
        break

    # write the frame
    # out.write(IPM_img)
    # if count==180:
    #     cv2.imwrite("implementacao/frame.png", frame)
    #     cv2.imwrite("implementacao/frame_bev.png", img_bev)
 
video.release()
cv2.destroyAllWindows()

###################################################################################################################

# angle = np.array(angle)*(180/np.pi)
# print(angle)
# angle_avg = median(angle)
# angle_stdev = pstdev(angle)
# print('média do ângulo =', angle_avg, 'desvio padrão do ângulo =', angle_stdev)
# frame_number = np.arange(0,len(angle),1)
# plt.figure(1)
# plt.plot(frame_number, angle, '-r')
# plt.xlabel('Frame')
# plt.ylabel('Ângulo entre faixas [°]')
# plt.grid()

# dist = np.array(dist)
# dist = 0.1*dist
# dist_avg = median(dist)
# dist_stdev = pstdev(dist)
# print('média da distância =', dist_avg, 'desvio padrão da distância =', dist_stdev)
# frame_number = np.arange(0,len(dist),1)
# plt.figure(2)
# plt.plot(frame_number, dist, '-r', )
# plt.xlabel('Frame')
# plt.ylabel('Distância entre faixas [m]')
# plt.grid()

# pitch = np.array(pitch)*(180/np.pi)
# pitch_avg = median(pitch)
# pitch_stdev = pstdev(pitch)
# print('média do ângulo pitch =', pitch_avg, 'desvio padrão do ângulo pitch =', pitch_stdev)
# frame_number = np.arange(0,len(pitch),1)
# plt.figure(3)
# plt.plot(frame_number, pitch, '-r')
# plt.xlabel('Frame')
# plt.ylabel('Pitch [°]')
# plt.grid()

# yaw = np.array(yaw)*(180/np.pi)
# yaw_avg = median(yaw)
# yaw_stdev = pstdev(yaw)
# print('média do ângulo yaw =', yaw_avg, 'desvio padrão do ângulo yaw =', yaw_stdev)
# frame_number = np.arange(0,len(yaw),1)
# plt.figure(4)
# plt.plot(frame_number, yaw, '-r')
# plt.xlabel('Frame')
# plt.ylabel('Yaw [°]')
# plt.grid()

# plt.show()



