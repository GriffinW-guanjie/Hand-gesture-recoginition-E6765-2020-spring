     // Your web app's Firebase configuration
  var firebaseConfig = {
    apiKey: "",
    authDomain: "",
    databaseURL: "",
    projectId: "",
    storageBucket: "",
    messagingSenderId: "",
    appId: "",
    measurementId: ""
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  firebase.analytics();
  var ref = firebase.database().ref();
  ref.on("child_added", function(snapshot){console.log(snapshot.val())});

    // encapsulation
    // get element by id
    function $(id) {
        return document.getElementById(id);
    }

    // create div
    function creatediv(className) {
        var div = document.createElement('div');
        div.className = className;
        return div;
    }
    var clock = null;
    var state = 0;
    var speed = 6;
    var flag = false;


    //start game
    function start() {
        if(!flag) {
            init();
        } else {
            alert('Game started, no need to start again！')
        }
    }

    /*
     *    init
     */
    function init() {
        flag = true;


        for (var i = 0; i < 4; i++) {
                createrow();
        }
        var con = $('con');
        ref.on("child_added", function(snapshot){
        var X = snapshot.val()['x'] + 40;
        var Y = - snapshot.val()['y'] + 542;

        console.log(X,Y);
        console.log(document.elementFromPoint(X, Y).className);
        judge(X, Y);
        });

/*
            for (var i = 0; i < 4; i++) {
                createrow();
            }

            // add onclick event
            $('main').onclick = function (ev) {
                ev = ev || event;
                console.log(ev.clientX);
                console.log(ev.clientY);
                judge(ev);
            }
*/
            // clock every 50ms call move()

            clock = window.setInterval('move()', 100);
        }


    // judge whether the click is on black

    function judge(X, Y){
        if (document.elementFromPoint(X, Y).className){
        if (document.elementFromPoint(X, Y).className == 'cell black'){
            document.elementFromPoint(X, Y).className = 'cell';
            document.elementFromPoint(X, Y).parentNode.pass = 1;
            score();
        }
        else {
            document.elementFromPoint(X, Y).parentNode.pass1 = 1;
        }}
    }

    /*
    function judge(ev) {
        if(ev.target.className.indexOf('black') == -1 && ev.target.className.indexOf('cell') !== -1) {
            ev.target.parentNode.pass1 = 1; //pass, represent the white has been clicked
        }

        if (ev.target.className.indexOf('black') !== -1) {//click on black
            ev.target.className = 'cell';
            ev.target.parentNode.pass = 1; //pass, represent the black has been clicked
            score();
        }


    }*/

// judge if the game is over

function over() {
    var rows = con.childNodes;
    if ((rows.length == 5) && (rows[rows.length - 1].pass !== 1)) {
        fail();
    }
    for(let i = 0; i < rows.length; i++){
        if(rows[i].pass1 == 1) {
            fail();
        }
    }

}

    // game over
    function fail() {
        clearInterval(clock);
        flag = false;
        confirm('Your final score ' + parseInt($('score').innerHTML));
        var con = $('con');
        con.innerHTML =  "";
        $('score').innerHTML = 0;
        con.style.top = '-408px';

        // clean the firebase
        ref.remove();

    }

    // create <div class="row"> and it has 4 <div class="cell">
    function createrow() {
        var con = $('con');
        var row = creatediv('row'); //create div className=row
        var arr = creatcell();

        con.appendChild(row); // append row as con's node

        for (var i = 0; i < 4; i++) {
            row.appendChild(creatediv(arr[i])); //add cell
        }

        if (con.firstChild == null) {
            con.appendChild(row);
        } else {
            con.insertBefore(row, con.firstChild);
        }
    }


    // create cells in a row
    function creatcell() {
        var temp = ['cell', 'cell', 'cell', 'cell', ];
        var i = Math.floor(Math.random() * 4);//randomly generate the position of black
        temp[i] = 'cell black';
        return temp;
    }

    //move the rows
    function move() {
        var con = $('con');
        var top = parseInt(window.getComputedStyle(con, null)['top']);

        if (speed + top > 0) {
            top = 0;
        } else {
            top += speed;
        }
        con.style.top = top + 'px';//move the top
        over();
        if (top == 0) {
            createrow();
            con.style.top = '-102px';
            delrow();
        }
    }


    // speed up
    function speedup() {
        speed += 2;
        if (speed == 20) {
            alert('Amazing!');
        }
    }

    //delete a row
    function delrow() {
        var con = $('con');
        if (con.childNodes.length == 6) {
            con.removeChild(con.lastChild);
        }
    }

    // record the score
    function score() {
        var newscore = parseInt($('score').innerHTML) + 1;
        $('score').innerHTML = newscore; // update score
        /*if (newscore % 10 == 0) {//every 10 rows/points, speed the game up
            speedup();
        }*/
    }