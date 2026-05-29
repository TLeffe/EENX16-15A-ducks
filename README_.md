
### ROSNoder

### 1 Comm_node
Har funktioner för att ta emot instruktioner via UDP på en specifik port via SOCKET, motsvarande sändare finns i  ros2_ws/src/ducks/ducks/duck_control_node.py.

Topic som skickar vidare heter /{self._vehicle_name}/Comm_node/instructions
på detta format
    Inkommande meddelanden förväntas komma på formen. 
    {duck_name},x,y,theta,x1,y1,x2,y2nd
    Där X Y är nuvarande postion.
    Vinkel är vilken “heading” botten har 
    X1 Y1 postion den ska
    X2 Y2 Position därefter.

Vi rekomenderar framtida projekt att använde en 2 vägs kommunikation av mågot slag, om möjligt ROS2 brygga. 

### 2 system_id_node
Användes för systemidentifiering av robotarna, dessa kör i en S-kurva och loggar tidsstämpel, önskad vinkelhastighet, rotation kring z, beräknad vinkelhastighet, rå vinkelhastighet från IMU. Detta sparas i en CSV som enkelt heter data.

Plocka ut data ni behöver om ni ska använda duckiebots, utför datainsamling på samma underlag som ni kommer köra på i slutändan.

### 3 Twist_control.
Här sköts majoriteten av körlogiken.
lyssnar på information från comm_node på /{self._vehicle_name}/Comm_node/instructions.
De enskilda funktionerna är förklarade i koden. 
Det finns många debugmeddelanden som kan kommenteras bort vid faktisk körning
Uppdateringsfrekvenser över 25hz leder ofta till att avstängningskommandot inte exekveras korrekt och manuell handpåläggning för att stänga av krävs. Detta kringår avstängningsprocessen och vi misstänker detta är kan leda till korruption av data. 

### 4 obstacle_detection
Hindredetektion användes för att undvika kollisioner. 
lyssnar på information från obstacle_detection på /{self.vehicle_name}/obstacle_detection_node/obstacle_detected. De enskilda funktioner är förklarade i koden. 
ToF-sensor fungerar utmärkt utan något problem, men bildanalysen behöver felsökans. 




### Att komma igång.

Börja med att följa setupguiden på duckietown och läsa 'duckumentation'.

Men de centralaste kommandon är 

dts devel build -H {name} -f
    För att bygga ny kod på given robot
dts devel run -H {name} -L {launcher_name}
    för att köra given launcher på given robot
dts fleet discover
    för att kolla vilka robotar som har går att hitta på nätet.

Viktigt att notera att du behöver vara på ett nätverk där det går att pinga/kommunicera obehindrat, med andra ord INTE eduroam, kontakta er handledare eller IT administrationen för att komma runt problemet. 

Ni kommer att behöva implementera egen styrning, den som finns här kan ni se som ett ramverk att arbeta vidare ifrån. För den fungerar.

### vanliga Errors och lösningar
Dessa är de vanligaste errors vi stött på, saker som att roboten slutar rotera och beter sig konstigt har inte behandlats här. Felsökningsguiden för duckiebots från duckietowns hemsida är också en bra plats att påbörja processen. 


### Error response from daemon: client version 1.52 is too new. Maximum supported API version is 1.41

    Om detta error dyker upp finns en temporär fix för att minska versionen koden körs i. Uppdaterar boten till en äldre version
    Kör:
    export DOCKER_API_VERSION=1.41
    För att ändra versionen du kör på 

### Problem med att hitta robot fastän allt är på nätverket

    Testa att istället för att köra {name} på roboten, använd robotens IP adress, denna ska stå på skärmen, bläddra genom ett snabbt tryck på knappen.

    Säkerställ att det inte finns någon brandvägg i vägen.

    Säkerställ att roboten inte har låst sig. 

    Säkerställ att ni är på rätt nätverk.

### "allt" slutar fungera.
Ibland vid felsökning verkar robotarna helt sluta fungera. Om ni kopplar in tangentbord och skärm till roboten vid uppstart kan ni se vad som sker vid startup. Detta kan iböand vara hjälpsamt för att se om den löser sig under setup. 

Men i absolut värsta fall, gör om hela setup för SD kortet. Detta tar några timmar men brukar lösa dessa problem. Vi tror att det är kopplat till att robotar ibland måste stängas av genom att  dra ur sladden, eftersom att de ibland låser sig. 



