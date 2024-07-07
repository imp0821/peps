WITH
  input_feature_map AS (
	SELECT id, (c.image).array_data AS value
	from cifar c),
  Conv506_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM input_feature_map I, Conv506_weight W
	),
  Conv512_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Conv506_fwd0 I, Conv512_weight W
	),	 
  Conv507_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Conv506_fwd0 I, Conv507_weight W
	),
  Conv508_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Conv506_fwd0 I, Conv508_weight W
	),
  Conv509_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Conv506_fwd0 I, Conv509_weight W
	),
  Conv510_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=5, padding:=2, stride:=1) AS value
	FROM Conv508_fwd0 I, Conv510_weight W
	),
  Conv511_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv509_fwd0 I, Conv511_weight W
	),
  Conv513_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv511_fwd0 I, Conv513_weight W
	),
  Concat302_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv507_fwd0 I1 
	INNER JOIN Conv510_fwd0 I2 on I1.id=I2.id
	INNER JOIN Conv513_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv512_fwd0 I4 on I1.id=I4.id
	),
  Conv519_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat302_fwd0 I, Conv519_weight W
	),	 
  Conv514_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat302_fwd0 I, Conv514_weight W
	),
  Conv515_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat302_fwd0 I, Conv515_weight W
	),
  Conv516_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat302_fwd0 I, Conv516_weight W
	),
  Conv517_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=5, padding:=2, stride:=1) AS value
	FROM Conv515_fwd0 I, Conv517_weight W
	),
  Conv518_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv516_fwd0 I, Conv518_weight W
	),
  Conv520_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv518_fwd0 I, Conv520_weight W
	),
  Concat319_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv514_fwd0 I1 
	INNER JOIN Conv517_fwd0 I2 on I1.id=I2.id
	INNER JOIN Conv520_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv519_fwd0 I4 on I1.id=I4.id
	),
  Conv526_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat319_fwd0 I, Conv526_weight W
	),	
  Conv521_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat319_fwd0 I, Conv521_weight W
	),
  Conv522_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat319_fwd0 I, Conv522_weight W
	),
  Conv523_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat319_fwd0 I, Conv523_weight W
	),
  Conv524_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=5, padding:=2, stride:=1) AS value
	FROM Conv522_fwd0 I, Conv524_weight W
	),
  Conv525_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv523_fwd0 I, Conv525_weight W
	),
  Conv527_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv525_fwd0 I, Conv527_weight W
	),
  Concat336_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv521_fwd0 I1 
	INNER JOIN Conv524_fwd0 I2 on I1.id=I2.id
	INNER JOIN Conv527_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv526_fwd0 I4 on I1.id=I4.id
	),
  MaxPool340_fwd0 AS (
	SELECT id, maxpool(I.value, kernel:=3, padding:=0, stride:=2) AS value
	FROM Concat336_fwd0 I
	),
  Conv528_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat336_fwd0 I, Conv528_weight W
	),
  Conv592_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=0, stride:=2) AS value
	FROM Concat336_fwd0 I, Conv592_weight W
	),
  Conv529_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv528_fwd0 I, Conv529_weight W
	),
  Conv593_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=0, stride:=2) AS value
	FROM Conv529_fwd0 I, Conv593_weight W
	),
  Concat347_fwd0 AS (
	SELECT I1.id, concat_array(3, input1:=I1.value, input2:=I2.value, input3:=I3.value) AS value
	FROM Conv592_fwd0 I1 
	INNER JOIN Conv593_fwd0 I2 on I1.id=I2.id
	INNER JOIN MaxPool340_fwd0 I3 on I1.id=I3.id
	),
  Conv535_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat347_fwd0 I, Conv535_weight W
	),	
  Conv530_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat347_fwd0 I, Conv530_weight W
	),
  Conv531_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat347_fwd0 I, Conv531_weight W
	),
  Conv532_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat347_fwd0 I, Conv532_weight W
	),
  Conv533_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv531_fwd0 I, Conv533_weight W
	),
  Conv534_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv532_fwd0 I, Conv534_weight W
	),
  Conv536_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv533_fwd0 I, Conv536_weight W
	),
  Conv537_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv534_fwd0 I, Conv537_weight W
	),
  Conv538_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv537_fwd0 I, Conv538_weight W
	),
  Conv539_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv538_fwd0 I, Conv539_weight W
	),
  Concat370_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv530_fwd0 I1 
	INNER JOIN Conv536_fwd0 I2 on I1.id=I2.id
	INNER JOIN Conv539_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv535_fwd0 I4 on I1.id=I4.id
	),
  Conv545_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat370_fwd0 I, Conv545_weight W
	),	
  Conv540_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat370_fwd0 I, Conv540_weight W
	),
  Conv541_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat370_fwd0 I, Conv541_weight W
	),
  Conv542_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat370_fwd0 I, Conv542_weight W
	),
  Conv543_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv541_fwd0 I, Conv543_weight W
	),
  Conv544_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv542_fwd0 I, Conv544_weight W
	),
  Conv546_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv543_fwd0 I, Conv546_weight W
	),
  Conv547_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv544_fwd0 I, Conv547_weight W
	),
  Conv548_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv547_fwd0 I, Conv548_weight W
	),
  Conv549_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv548_fwd0 I, Conv549_weight W
	),
  Concat393_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv540_fwd0 I1 
	INNER JOIN Conv546_fwd0 I2 on I1.id=I2.id
	INNER JOIN Conv549_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv545_fwd0 I4 on I1.id=I4.id
	),
  Conv555_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat393_fwd0 I, Conv555_weight W
	),	
  Conv550_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat393_fwd0 I, Conv550_weight W
	),
  Conv551_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat393_fwd0 I, Conv551_weight W
	),
  Conv552_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat393_fwd0 I, Conv552_weight W
	),
  Conv553_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv551_fwd0 I, Conv553_weight W
	),
  Conv554_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv552_fwd0 I, Conv554_weight W
	),
  Conv556_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv553_fwd0 I, Conv556_weight W
	),
  Conv557_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv554_fwd0 I, Conv557_weight W
	),
  Conv558_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv557_fwd0 I, Conv558_weight W
	),
  Conv559_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv558_fwd0 I, Conv559_weight W
	),
  Concat416_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv550_fwd0 I1 
	INNER JOIN Conv556_fwd0 I2 on I1.id=I2.id
	INNER JOIN Conv559_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv555_fwd0 I4 on I1.id=I4.id
	),
  Conv565_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat416_fwd0 I, Conv565_weight W
	),	
  Conv560_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat416_fwd0 I, Conv560_weight W
	),
  Conv561_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat416_fwd0 I, Conv561_weight W
	),
  Conv562_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat416_fwd0 I, Conv562_weight W
	),
  Conv563_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv561_fwd0 I, Conv563_weight W
	),
  Conv564_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv562_fwd0 I, Conv564_weight W
	),
  Conv566_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv563_fwd0 I, Conv566_weight W
	),
  Conv567_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv564_fwd0 I, Conv567_weight W
	),
  Conv568_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv567_fwd0 I, Conv568_weight W
	),
  Conv569_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv568_fwd0 I, Conv569_weight W
	),
  Concat439_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv560_fwd0 I1 
	INNER JOIN Conv566_fwd0 I2 on I1.id=I2.id
	INNER JOIN Conv569_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv565_fwd0 I4 on I1.id=I4.id
	),
  MaxPool443_fwd0 AS (
	SELECT id, maxpool(I.value, kernel:=3, padding:=0, stride:=2) AS value
	FROM Concat439_fwd0 I
	),
  Conv570_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat439_fwd0 I, Conv570_weight W
	),
  Conv571_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat439_fwd0 I, Conv571_weight W
	),
  Conv594_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=0, stride:=2) AS value
	FROM Conv570_fwd0 I, Conv594_weight W
	),
  Conv572_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 1, 7, 0, 3, 1
		) AS value
	FROM Conv571_fwd0 I, Conv572_weight W
	),
  Conv573_fwd0 AS (
	SELECT id, kfm_im2col_ns(
		W.val, I.value, W.bias, 7, 1, 3, 0, 1
		) AS value
	FROM Conv572_fwd0 I, Conv573_weight W
	),
  Conv595_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=0, stride:=2) AS value
	FROM Conv573_fwd0 I, Conv595_weight W
	),
  Concat454_fwd0 AS (
	SELECT I1.id, concat_array(3, input1:=I1.value, input2:=I2.value, input3:=I3.value) AS value
	FROM Conv594_fwd0 I1 
	INNER JOIN Conv595_fwd0 I2 on I1.id=I2.id
	INNER JOIN MaxPool443_fwd0 I3 on I1.id=I3.id
	),
  Conv580_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat454_fwd0 I, Conv580_weight W
	),	
  Conv574_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat454_fwd0 I, Conv574_weight W
	),
  Conv575_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat454_fwd0 I, Conv575_weight W
	),
  Concat471_fwd0 AS (
	SELECT id, concat_array(2, 
	input1:=kfm_im2col_ns(
		W1.val, I.value, W1.bias, 1, 3, 0, 1, 1
		),
	input2:=kfm_im2col_ns(
		W2.val, I.value, W2.bias, 3, 1, 1, 0, 1
		) 
	) as value
	FROM Conv575_fwd0 I, Conv577_weight W1, Conv578_weight W2
	),
  Conv576_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat454_fwd0 I, Conv576_weight W
	),
  Conv579_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv576_fwd0 I, Conv579_weight W
	),
  Concat476_fwd0 AS (
	SELECT id, concat_array(2,
	input1:=kfm_im2col_ns(
		W1.val, I.value, W1.bias, 1, 3, 0, 1, 1
		),
	input2:=kfm_im2col_ns(
		W2.val, I.value, W2.bias, 3, 1, 1, 0, 1
		) 
	) as value
	FROM Conv579_fwd0 I, Conv581_weight W1, Conv582_weight W2
	),
  Concat477_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv574_fwd0 I1 
	INNER JOIN Concat471_fwd0 I2 on I1.id=I2.id
	INNER JOIN Concat476_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv580_fwd0 I4 on I1.id=I4.id
	),
  Conv589_fwd0 AS (
	SELECT id, avgpool_conv(W.val, I.value, W.bias, 
					kernel:=3, padding:=1, stride:=1) as value
	FROM Concat477_fwd0 I, Conv589_weight W
	),	
  Conv583_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat477_fwd0 I, Conv583_weight W
	),
  Conv584_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat477_fwd0 I, Conv584_weight W
	),
  Conv585_fwd0 AS (
	SELECT id, kfm_nt(W.val, I.value, W.bias) AS value
	FROM Concat477_fwd0 I, Conv585_weight W
	),
  Concat494_fwd0 AS (
	SELECT id, concat_array(2,
	input1:=kfm_im2col_ns(
		W1.val, I.value, W1.bias, 1, 3, 0, 1, 1
		),
	input2:=kfm_im2col_ns(
		W2.val, I.value, W2.bias, 3, 1, 1, 0, 1
		)
	) as value
	FROM Conv584_fwd0 I, Conv586_weight W1, Conv587_weight W2
	),
  Conv588_fwd0 AS (
	SELECT id, kfm_im2col(W.val, I.value, W.bias, kernel:=3, padding:=1, stride:=1) AS value
	FROM Conv585_fwd0 I, Conv588_weight W
	),
  Concat499_fwd0 AS (
	SELECT id, concat_array(2, 
	input1:=kfm_im2col_ns(
		W1.val, I.value, W1.bias, 1, 3, 0, 1, 1
		),
	input2:=kfm_im2col_ns(
		W2.val, I.value, W2.bias, 3, 1, 1, 0, 1
		) 
	) as value
	FROM Conv588_fwd0 I, Conv590_weight W1, Conv591_weight W2
	),
  Concat500_fwd0 AS (
	SELECT I1.id, concat_array(4, input1:=I1.value, input2:=I2.value, input3:=I3.value, input4:=I4.value) AS value
	FROM Conv583_fwd0 I1 
	INNER JOIN Concat494_fwd0 I2 on I1.id=I2.id
	INNER JOIN Concat499_fwd0 I3 on I1.id=I3.id
	INNER JOIN Conv589_fwd0 I4 on I1.id=I4.id
	),
  AveragePool502_fwd0 AS (
	SELECT id, avgpool(I.value, kernel:=7, padding:=0, stride:=7) AS value
	FROM Concat500_fwd0 I
	),
  Reshape503_fwd0 AS (
	SELECT id, array_agg(t.value) as value
	FROM (SELECT id, unnest(I.value) as value FROM AveragePool502_fwd0 I) AS t
	GROUP BY id
	),
  MatMul505_fwd0 AS (
	SELECT id, mvm(W.val, I.value) AS value
	FROM Reshape503_fwd0 I, Transpose504_input W
	)
SELECT t.id, l.name AS res FROM
cifar10_labels l
JOIN (
	SELECT id, argmax(value) AS label FROM MatMul505_fwd0) t
	ON t.label = l.label