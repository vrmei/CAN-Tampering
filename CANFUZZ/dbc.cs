using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace PCANBasicExample
{

    public partial class dbc : Form
    {
        public dbc()
        {
            InitializeComponent();
        }
        public string dbcfiletext;
        public List<ListViewData> listViewData = new List<ListViewData>();
        public Dictionary<CMsg, List<CSignal>> gmapMessageNameToSignal = new Dictionary<CMsg, List<CSignal>>();
        public Regex regexForMessage = new Regex("BO_ (\\d+) ([a-zA-z_0-9]+): [0-8] ([a-zA-Z]+)");
        public Regex regexForSignal = new Regex("SG_ ([a-zA-Z_0-9]+) : (\\d+)\\|(\\d+)@([01])([\\+-]) \\(([\\d.-]+),([\\d.-]+)\\) \\[([\\d.-]+)\\|([\\d.-]+)\\][ ]+\\\"([a-zA-Z0-9/^]*)\\\"[ ]+([a-zA-Z_,0-9]+)");
        public void changetext()
        {
            List<string> vecLine = new List<string>(dbcfiletext.Split('\n'));
            for (int i = 0; i < vecLine.Count; i++)
            {
                List<CSignal> tmpvecSignal = new List<CSignal>();
                CSignal tmpSignal;
                Match matchMessage;
                Match matchSignal;
                bool bRet = regexForMessage.IsMatch(vecLine[i]);
                if (bRet)
                {
                    matchMessage = regexForMessage.Match(vecLine[i]);
                    while (i + 1 < vecLine.Count && regexForSignal.IsMatch(vecLine[i + 1]))
                    {
                        matchSignal = regexForSignal.Match(vecLine[i + 1]);
                        tmpSignal = new CSignal
                        {
                            szSignalName = matchSignal.Groups[1].Value,
                            startbit = int.Parse(matchSignal.Groups[2].Value),
                            length = int.Parse(matchSignal.Groups[3].Value),
                            bigorsmall = int.Parse(matchSignal.Groups[4].Value) == 0,
                            szType = matchSignal.Groups[5].Value == "+" ? "Unsigned" : "Signed",
                            dFactor = double.Parse(matchSignal.Groups[6].Value),
                            szUnit = matchSignal.Groups[10].Value
                        };
                        tmpvecSignal.Add(tmpSignal);
                        i++;
                    }
                    CMsg cMsg = new CMsg();
                    cMsg.CANID = int.Parse(matchMessage.Groups[1].Value);
                    cMsg.szMessageName = matchMessage.Groups[2].Value;
                    gmapMessageNameToSignal.Add(cMsg, tmpvecSignal);
                }
            }
            // 插入列
            list_dbc.Columns.Add("", 0, HorizontalAlignment.Left);
            list_dbc.Columns.Add("Message name", 300, HorizontalAlignment.Left);
            list_sig.Columns.Add("", 0, HorizontalAlignment.Left);
            list_sig.Columns.Add("Signal name", 400, HorizontalAlignment.Left);
            list_sig.Columns.Add("Start bit", 150, HorizontalAlignment.Left);
            list_sig.Columns.Add("Length [Bit]", 200, HorizontalAlignment.Left);
            list_sig.Columns.Add("Byte Order", 200, HorizontalAlignment.Left);
            list_sig.Columns.Add("Value Type", 200, HorizontalAlignment.Left);
            list_payload.Columns.Add("", 0, HorizontalAlignment.Left);
            list_payload.Columns.Add("Message name", 150, HorizontalAlignment.Center);
            list_payload.Columns.Add("CAN ID", 100, HorizontalAlignment.Center);
            list_payload.Columns.Add("频率(Hz)", 100, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 0", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 1", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 2", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 3", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 4", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 5", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 6", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("data 7", 70, HorizontalAlignment.Center);
            list_payload.Columns.Add("持续时间(s)", 150, HorizontalAlignment.Center);
            // 插入项
            int l = 0;
            foreach (var item in gmapMessageNameToSignal)
            {
                string tmpStr = item.Key.szMessageName;
                ListViewItem listViewItem = new ListViewItem(new[] { "", tmpStr });
                list_dbc.Items.Add(listViewItem);
                l++;
            }
        }
        public class CSignal
        {
            public string szSignalName;
            public int startbit;
            public int length;
            public bool bigorsmall;
            public string szType;
            public double dFactor;
            public string szUnit;
        }
        public class CMsg
        {
            public string szMessageName;
            public int CANID;
        }

        private void list_dbc_ItemSelectionChanged(object sender, ListViewItemSelectionChangedEventArgs e)
        {
            list_sig.Items.Clear(); // 清除list_sig的所有项

            // 遍历list_dbc中的每一项，查找选中的项
            foreach (ListViewItem item in list_dbc.SelectedItems)
            {
                string stmpMessageName = item.SubItems[1].Text;
                List<CSignal> signals = new List<CSignal>();
                foreach (var key in gmapMessageNameToSignal.Keys)
                {
                    string MSG = key.szMessageName;
                    if(MSG == stmpMessageName)
                    {
                        signals = gmapMessageNameToSignal[key];
                    }
                }

                if (signals.Count == 0)
                {
                    list_sig.Items.Add(new ListViewItem(new string[] { "", "...<None>" }));
                    continue;
                }

                // 对信号列表进行排序（需要实现IComparable<CSignal>接口或提供比较器）
                //signals.Sort();

                for (int j = 0; j < signals.Count; j++)
                {
                    CSignal tmpSignal = signals[j];
                    var listViewItem = new ListViewItem("");
                    listViewItem.SubItems.Add(tmpSignal.szSignalName);
                    listViewItem.SubItems.Add(tmpSignal.startbit.ToString());
                    listViewItem.SubItems.Add(tmpSignal.length.ToString());
                    listViewItem.SubItems.Add(tmpSignal.bigorsmall ? "Motorola" : "Intel");
                    listViewItem.SubItems.Add(tmpSignal.szType);
                    list_sig.Items.Add(listViewItem);
                }
            }
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = sender as CheckBox;

            if (checkBox != null)
            {
                switch (checkBox.CheckState)
                {
                    case CheckState.Checked:
                        checkBox1.Enabled = false;
                        comboBox2.Enabled = false;
                        textBox1.Enabled = true;
                        break;
                    case CheckState.Unchecked:
                        checkBox1.Checked = true;
                        checkBox1.Enabled = true;
                        checkBox2.Enabled = false;
                        textBox1.Enabled = false;
                        comboBox2.Enabled = true;
                        break;
                    case CheckState.Indeterminate:
                        MessageBox.Show("复选框状态不确定！");
                        break;
                }
            }
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = sender as CheckBox;

            if (checkBox != null)
            {
                switch (checkBox.CheckState)
                {
                    case CheckState.Checked:
                        checkBox2.Enabled = false;
                        comboBox2.Enabled = true;
                        textBox1.Enabled = false;
                        break;
                    case CheckState.Unchecked:
                        checkBox2.Checked = true;
                        checkBox1.Enabled = false;
                        checkBox2.Enabled = true;
                        textBox1.Enabled = true;
                        comboBox2.Enabled = false;
                        break;
                    case CheckState.Indeterminate:
                        MessageBox.Show("复选框状态不确定！");
                        break;
                }
            }
        }

        private void list_dbc_DoubleClick(object sender, EventArgs e)
        {
            foreach (ListViewItem item in list_dbc.SelectedItems)
            {
                string stmpMessageName = item.SubItems[1].Text;
                CMsg msg = new CMsg();
                foreach (var key in gmapMessageNameToSignal.Keys)
                {
                    string MSG = key.szMessageName;
                    if (MSG == stmpMessageName)
                    {
                        msg = key;
                    }
                }

                if (msg.szMessageName == "")
                {
                    list_payload.Items.Add(new ListViewItem(new string[] { "", "...<None>" }));
                    continue;
                }

                // 对信号列表进行排序（需要实现IComparable<CSignal>接口或提供比较器）
                //signals.Sort();
                var listViewItem = new ListViewItem("");
                listViewItem.SubItems.Add(msg.szMessageName);
                listViewItem.SubItems.Add(msg.CANID.ToString("X2"));
                string temp = comboBox2.Text;
                if (temp.Length <= 2)
                {
                    listViewItem.SubItems.Add("0");
                }
                else
                {
                    listViewItem.SubItems.Add(temp.Substring(0, temp.Length - 2));
                }
                if (checkBox3.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox4.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox5.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox6.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox7.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox8.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox9.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox10.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }

                string sztime = comboBox1.Text;
                if (sztime.Length <= 1)
                {  listViewItem.SubItems.Add("300"); }
                else { listViewItem.SubItems.Add(sztime.Substring(0, sztime.Length - 1)); }
                list_payload.Items.Add(listViewItem);

            }
        }

        private void Button_all_Click(object sender, EventArgs e)
        {
            checkBox3.Checked = !checkBox3.Checked;
            checkBox4.Checked = !checkBox4.Checked;
            checkBox5.Checked = !checkBox5.Checked;  
            checkBox6.Checked = !checkBox6.Checked;
            checkBox7.Checked = !checkBox7.Checked;
            checkBox8.Checked = !checkBox8.Checked;
            checkBox9.Checked = !checkBox9.Checked;
            checkBox10.Checked = !checkBox10.Checked;
        }

        private void random_Click(object sender, EventArgs e)
        {
            foreach (var msg in gmapMessageNameToSignal.Keys)
            {
                var listViewItem = new ListViewItem("");
                listViewItem.SubItems.Add(msg.szMessageName);
                listViewItem.SubItems.Add(msg.CANID.ToString("X2"));
                string temp = comboBox2.Text;
                if (temp.Length <= 2)
                {
                    listViewItem.SubItems.Add("0");
                }
                else
                {
                    listViewItem.SubItems.Add(temp.Substring(0, temp.Length - 2));
                }
                if (checkBox3.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox4.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox5.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox6.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox7.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox8.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox9.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }
                if (checkBox10.Checked)
                {
                    listViewItem.SubItems.Add("?");
                }
                else
                {
                    listViewItem.SubItems.Add("00");
                }

                string sztime = comboBox1.Text;
                if (sztime.Length <= 1)
                { listViewItem.SubItems.Add("300"); }
                else { listViewItem.SubItems.Add(sztime.Substring(0, sztime.Length - 1)); }
                list_payload.Items.Add(listViewItem);
            }
        }

        public class ListViewData
        {
            public string MessageName { get; set; }
            public string CanId { get; set; }
            public string Frequency { get; set; }
            public string Data0 { get; set; }
            public string Data1 { get; set; }
            public string Data2 { get; set; }
            public string Data3 { get; set; }
            public string Data4 { get; set; }
            public string Data5 { get; set; }
            public string Data6 { get; set; }
            public string Data7 { get; set; }
            public string Duration { get; set; }
        }
        private void send_Click(object sender, EventArgs e)
        {
            listViewData = new List<ListViewData>();

            foreach (ListViewItem item in list_payload.Items)
            {
                var data = new ListViewData
                {
                    MessageName = item.SubItems[1].Text,
                    CanId = item.SubItems[2].Text,
                    Frequency = item.SubItems[3].Text,
                    Data0 = item.SubItems[4].Text,
                    Data1 = item.SubItems[5].Text,
                    Data2 = item.SubItems[6].Text,
                    Data3 = item.SubItems[7].Text,
                    Data4 = item.SubItems[8].Text,
                    Data5 = item.SubItems[9].Text,
                    Data6 = item.SubItems[10].Text,
                    Data7 = item.SubItems[11].Text,
                    Duration = item.SubItems[12].Text
                };
                listViewData.Add(data);
            }
            this.Close();
        }
    }
}
