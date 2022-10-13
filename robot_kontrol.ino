#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;
int xx,t1,t2;
int sayac=0;
int aci1, aci1a=90;
int aci2, aci2a=60;
int aci3, aci3a=60;
int aci4, aci4a=60;
int aci5=60;
int pos = 0;
String gelen;
int time_f=1;
String data="";
unsigned long myTimeStart,myTimeStop ;
void setup() {
  // put your setup code here, to run once:
  
  //delay(1000);
  Serial.begin(115200);
  Serial.setTimeout(1);
 
}

void loop() {
   if (Serial.available() && sayac==0){
    String hazir = Serial.readString();
    Serial.print(hazir);
    sayac=1;
    Braccio.begin();
    unsigned long dd= millis();
   }
   if (Serial.available() && sayac==1){
    
    String gelen = Serial.readString();
    if (time_f ==1){
      myTimeStart = millis();}
      int t1=myTimeStart;
      time_f=5;
      
    //delay(250);
    //Serial.println(gelen);
    data = data + gelen;
    //Serial.print(data);
    if (data.length()==29){
      
      sayac=2;
      int aci1 = data.substring(0,3).toInt();
      int aci2 = data.substring(3,6).toInt();
      int aci3 = data.substring(6,9).toInt();
      int aci4 = data.substring(9,12).toInt();
      int aci5 = data.substring(12,15).toInt();
      int z1 = data.substring(15,20).toInt();
      int aci2a = data.substring(20,23).toInt();
      int aci3a = data.substring(23,26).toInt();
      int aci4a = data.substring(26).toInt();
      //myTimeStop=millis();
     // int t2=myTimeStop;
     // xx= z1 -(t2-t1);
      
      delay(100);
      Braccio.ServoMovement(20,         aci1a, aci2a, aci3a, aci4a, aci5,  10); 
      delay(10);
      Braccio.ServoMovement(20,         aci1, aci2a, aci3a, aci4a, aci5,  10); 
        delay(10);
        myTimeStop=millis();
         int t2=myTimeStop;
         Serial.print(t2-t1);
        int timm= long(z1)-(t2-t1);
        delay (timm);
        
        Braccio.ServoMovement(10,         aci1, aci2, aci3, aci4, aci5,  63); 
        delay(10);
        Braccio.ServoMovement(20,         aci1, aci2, 90, 90, 90,  63);  
        delay (2000);
        Braccio.ServoMovement(20,         aci1, aci2, 120, 120, 120,  10);
        delay (1000);
        //Braccio.ServoMovement(20,         90, 90, 135, 45, 60,  10);
        //delay (1000);
    //String hazir = Serial.readString();
    //Serial.println(aci2);
    //Serial.println(aci3);
    //Serial.println(aci4);
    //Serial.println(aci5);
    }
   }
    if(sayac==2){
        
        sayac = 1;
        gelen = "";
        data = "";
        time_f=1;
    }
//}
}
