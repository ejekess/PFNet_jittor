import re
from tensorboardX import SummaryWriter

def parse_log_and_convert_to_tensorboard(log_file_path, output_dir):
    """
    解析trina中保存的txt格式的训练log并转换为tensorboard格式

    Args:
        log_file_path: txt日志文件路径
        output_dir: tensorboard文件输出目录
    """


    writer = SummaryWriter(log_dir=output_dir, comment='converted_from_txt')

    # 正则表达式匹配日志格式
    # [epoch], [curr_iter], [base_lr], [loss], [loss_1], [loss_2], [loss_3], [loss_4]
    log_pattern = r'\[\s*(\d+)\],\s*\[\s*(\d+)\],\s*\[([\d\.e\-\+]+)\],\s*\[([\d\.e\-\+]+)\],\s*\[([\d\.e\-\+]+)\],\s*\[([\d\.e\-\+]+)\],\s*\[([\d\.e\-\+]+)\],\s*\[([\d\.e\-\+]+)\]'

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"开始解析日志文件: {log_file_path}")
        print(f"总行数: {len(lines)}")

        parsed_count = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # 跳过空行和非日志行
            if not line or not line.startswith('['):
                continue


            match = re.match(log_pattern, line)

            if match:
                epoch = int(match.group(1))
                curr_iter = int(match.group(2))
                lr = float(match.group(3))
                loss = float(match.group(4))
                loss_1 = float(match.group(5))
                loss_2 = float(match.group(6))
                loss_3 = float(match.group(7))
                loss_4 = float(match.group(8))

                # 写入tensorboard
                writer.add_scalar('learning_rate', lr, curr_iter)
                writer.add_scalar('loss_a/loss_avg', loss, curr_iter)
                writer.add_scalar('loss_a/loss1_avg', loss_1, curr_iter)
                writer.add_scalar('loss_a/loss2_avg', loss_2, curr_iter)
                writer.add_scalar('loss_a/loss3_avg', loss_3, curr_iter)
                writer.add_scalar('loss_a/loss4_avg', loss_4, curr_iter)

                # 按epoch分组的损失
                writer.add_scalar('epoch_loss/loss_avg', loss, epoch)
                writer.add_scalar('epoch_loss/loss_1_avg', loss_1, epoch)
                writer.add_scalar('epoch_loss/loss_2_avg', loss_2, epoch)
                writer.add_scalar('epoch_loss/loss_3_avg', loss_3, epoch)
                writer.add_scalar('epoch_loss/loss_4_avg', loss_4, epoch)

                parsed_count += 1

                if parsed_count % 100 == 0:
                    print(f"已解析 {parsed_count} 条记录...")

            else:

                print(f"第 {line_num} 行格式不匹配: {line}")

        writer.close()
        print(f"总记录数{parsed_count}")

    except Exception as e:
        print(f"解析过程中出现错误: {str(e)}")
        writer.close()





log_txt = './ckpt_jittor/PFNet/2025-07-18-12-38-37-217476.txt'
out_log = './ckpt_jittor/PFNet/log'


parse_log_and_convert_to_tensorboard(log_txt, out_log)

