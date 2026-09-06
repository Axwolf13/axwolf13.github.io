/* Live yoga pose demo.
   MoveNet Lightning (TF.js pose-detection) feeds 17 keypoints into the
   classifier head trained for the Sādhanā paper (11 asanas). The head's
   weights were extracted from the paper's TFLite file and run here as plain
   JavaScript (hip-centered normalization + 3 dense layers), verified to
   match the TFLite interpreter on all 571 held-out test samples.
   Everything runs client-side; no frames leave the browser. */
(function(){
  'use strict';

  var LABELS = [
    {sa:'Adho Mukha Svanasana', en:'Downward Dog'},
    {sa:'Balasana',             en:'Child'},
    {sa:'Bhujangasana',         en:'Cobra'},
    {sa:'Phalakasana',          en:'Plank'},
    {sa:'Setu Bandha Sarvangasana', en:'Bridge'},
    {sa:'Utkata Konasana',      en:'Goddess'},
    {sa:'Utkatasana',           en:'Chair'},
    {sa:'Virabhadrasana 1',     en:'Warrior 1'},
    {sa:'Virabhadrasana 2',     en:'Warrior 2'},
    {sa:'Virabhadrasana 3',     en:'Warrior 3'},
    {sa:'Vrikshasana',          en:'Tree'}
  ];

  // COCO-17 skeleton pairs, same convention as the paper's drawing code
  var EDGES = [[0,1],[0,2],[1,3],[2,4],[5,7],[7,9],[6,8],[8,10],
               [5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];

  // First row of the paper's train_data.csv (Downward Dog). Used once after
  // the weights load to confirm the wiring matches training.
  var SELF_TEST = [659,535,0.7139464,635,544,0.77138937,635,539,0.70211506,
    589,510,0.79435617,587,508,0.82762396,602,422,0.64859605,603,416,0.55352294,
    474,573,0.6860989,477,549,0.7249965,370,638,0.6707754,372,624,0.81982476,
    850,164,0.8632088,838,171,0.8183702,1002,400,0.89810854,972,412,0.8311561,
    1133,624,0.820135,1083,604,0.8374676];

  var TORSO = [5,6,11,12];        // shoulders + hips
  var LEGS  = [13,14,15,16];      // knees + ankles
  var TORSO_MIN  = 0.2;
  var LEGS_MIN   = 0.15;
  var DRAW_SCORE = 0.3;           // gate for drawing a keypoint/edge
  var SMOOTH_N   = 8;             // frames of probability averaging
  var SHOW_CONF  = 0.5;           // smoothed probability needed for a firm label

  var video    = document.getElementById('cam');
  var canvas   = document.getElementById('stage');
  var ctx      = canvas.getContext('2d');
  var stageMsg = document.getElementById('stageMsg');
  var startBtn = document.getElementById('startBtn');
  var stopBtn  = document.getElementById('stopBtn');
  var statusEl = document.getElementById('status');
  var fpsEl    = document.getElementById('fps');
  var nameEl   = document.getElementById('asanaName');
  var subEl    = document.getElementById('asanaSub');
  var confFill = document.getElementById('confFill');
  var confLbl  = document.getElementById('confLabel');
  var chipsEl  = document.getElementById('chips');

  var detector = null, weights = null, stream = null;
  var classifierError = '';
  var running = false, rafId = null;
  var probHistory = [];
  var lastT = 0, fpsAvg = 0;
  var chips = [];
  var debugHud = false;
  var dbg = {minAll:0, gate:'', top3:''};

  LABELS.forEach(function(l){
    var c = document.createElement('span');
    c.className = 'chip';
    c.textContent = l.en;
    c.title = l.sa;
    chipsEl.appendChild(c);
    chips.push(c);
  });

  function setStatus(t){ statusEl.textContent = t; }

  function accent(){
    return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#11A05A';
  }

  /* --- classifier: normalization + 3 dense layers, ported from the paper's
         TFLite graph. Input: flat [x,y,score] x 17 in image pixels. --- */
  function dense(x, W, b, len, inLen){
    var out = new Float32Array(len);
    for(var i = 0; i < len; i++){
      var s = b[i], row = W[i];
      for(var j = 0; j < inLen; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }
  function relu6(x){
    for(var i = 0; i < x.length; i++) x[i] = Math.min(Math.max(x[i], 0), 6);
    return x;
  }
  function embedVec(vec51){
    // keep x,y only; center on the hip midpoint
    var xs = new Float32Array(17), ys = new Float32Array(17);
    for(var i = 0; i < 17; i++){ xs[i] = vec51[i*3]; ys[i] = vec51[i*3+1]; }
    var cx = (xs[11] + xs[12]) / 2, cy = (ys[11] + ys[12]) / 2;
    for(i = 0; i < 17; i++){ xs[i] -= cx; ys[i] -= cy; }
    // torso size: shoulder midpoint to hip midpoint (hips are now at origin)
    var sx = (xs[5] + xs[6]) / 2, sy = (ys[5] + ys[6]) / 2;
    var torso = Math.sqrt(sx*sx + sy*sy);
    // max dist as in the training graph: max of the column norms
    var nx = 0, ny = 0;
    for(i = 0; i < 17; i++){ nx += xs[i]*xs[i]; ny += ys[i]*ys[i]; }
    var maxDist = Math.max(Math.sqrt(nx), Math.sqrt(ny));
    var poseSize = Math.max(torso * 2.5, maxDist) || 1;
    var emb = new Float32Array(34);
    for(i = 0; i < 17; i++){ emb[i*2] = xs[i] / poseSize; emb[i*2+1] = ys[i] / poseSize; }
    return emb;
  }
  function classifyEmb(emb){
    var h = relu6(dense(emb, weights.W1, weights.b1, 128, 34));
    h = relu6(dense(h, weights.W2, weights.b2, 64, 128));
    var z = dense(h, weights.W3, weights.b3, 11, 64);
    var max = -Infinity, sum = 0, probs = new Float32Array(11);
    var i;
    for(i = 0; i < 11; i++) if(z[i] > max) max = z[i];
    for(i = 0; i < 11; i++){ probs[i] = Math.exp(z[i] - max); sum += probs[i]; }
    for(i = 0; i < 11; i++) probs[i] /= sum;
    return probs;
  }
  function classifyVec(vec51){ return classifyEmb(embedVec(vec51)); }

  /* Form: distance from the training set's average geometry for the detected
     pose (two reference centroids per class to cover mirrored variants).
     1.0 = as close as a typical training image, 0.5 = as far as the worst 5%.
     A rough typicality hint, not coaching. */
  function formScore(emb, idx){
    var best = Infinity;
    weights.centroids[idx].forEach(function(c){
      var s = 0;
      for(var j = 0; j < 34; j++){ var d = emb[j] - c[j]; s += d * d; }
      if(s < best) best = s;
    });
    var dist = Math.sqrt(best);
    var d50 = weights.d50[idx];
    var span = Math.max(weights.p95[idx] - d50, 1e-6);
    return Math.exp(-Math.LN2 * Math.max(dist - d50, 0) / span);
  }
  function formBand(f){
    return f >= 0.75 ? 'solid' : (f >= 0.4 ? 'okay' : 'loose');
  }

  async function loadModels(){
    setStatus('loading pose model…');
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING}
    );
    setStatus('loading classifier…');
    try{
      var resp = await fetch('classifier_weights.json');
      if(!resp.ok) throw new Error('HTTP ' + resp.status);
      weights = await resp.json();
      var p = classifyVec(SELF_TEST);
      var top = 0;
      for(var i = 1; i < 11; i++) if(p[i] > p[top]) top = i;
      if(top !== 0 || p[top] < 0.5){
        classifierError = 'self-test failed';
        console.warn('Classifier self-test expected Downward Dog >50%, got',
                     LABELS[top].en, p[top], '- disabling classification.');
        weights = null;
      }
    }catch(e){
      classifierError = 'failed to load (' + (e && e.message ? e.message.slice(0, 80) : 'unknown') + ')';
      console.warn('Classifier unavailable, keypoints-only mode:', e);
      weights = null;
    }
  }

  async function start(){
    startBtn.disabled = true;
    setStatus('requesting camera…');
    try{
      stream = await navigator.mediaDevices.getUserMedia({
        video: {width: {ideal: 640}, height: {ideal: 480}, facingMode: 'user'},
        audio: false
      });
    }catch(e){
      setStatus('camera blocked. Allow camera access and try again.');
      startBtn.disabled = false;
      return;
    }
    video.srcObject = stream;
    await video.play();
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    try{
      if(!detector) await loadModels();
    }catch(e){
      console.error(e);
      setStatus('model failed to load. Check your connection and reload.');
      stop();
      return;
    }
    stageMsg.hidden = true;
    stopBtn.hidden = false;
    running = true;
    probHistory = [];
    formHistory = [];
    if(weights){
      setStatus('tracking · classifier ready');
    }else{
      setStatus('tracking · keypoints only');
      nameEl.textContent = '—';
      subEl.textContent = 'classifier ' + (classifierError || 'unavailable') + '; keypoint tracking only';
    }
    lastT = performance.now();
    rafId = requestAnimationFrame(loop);
  }

  function stop(){
    running = false;
    if(rafId) cancelAnimationFrame(rafId);
    if(stream){ stream.getTracks().forEach(function(tr){ tr.stop(); }); stream = null; }
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stageMsg.hidden = false;
    stopBtn.hidden = true;
    startBtn.disabled = false;
    fpsEl.textContent = '';
    setStatus('idle');
    nameEl.textContent = '—';
    subEl.textContent = 'start the camera and hold a pose';
    confFill.style.width = '0';
    confLbl.textContent = '';
    chips.forEach(function(c){ c.classList.remove('active'); });
  }

  function groupMin(kps, idxs){
    var m = 1;
    idxs.forEach(function(i){ if(kps[i].score < m) m = kps[i].score; });
    return m;
  }

  function jointAngle(a, b, c){
    var ux = a.x - b.x, uy = a.y - b.y, vx = c.x - b.x, vy = c.y - b.y;
    var cos = (ux*vx + uy*vy) / (Math.hypot(ux, uy) * Math.hypot(vx, vy) || 1);
    return Math.acos(Math.min(Math.max(cos, -1), 1)) * 180 / Math.PI;
  }

  /* The classifier has no "not an asana" class, so a neutral stand maps to
     the nearest pose (Tree). Catch it geometrically: upright torso, straight
     knees, both feet grounded, arms hanging. No asana in the set fits that. */
  function isNeutralStand(k){
    var shoulderMid = {x:(k[5].x+k[6].x)/2, y:(k[5].y+k[6].y)/2};
    var hipMid      = {x:(k[11].x+k[12].x)/2, y:(k[11].y+k[12].y)/2};
    var torsoLen = Math.hypot(shoulderMid.x-hipMid.x, shoulderMid.y-hipMid.y) || 1;
    var upright = shoulderMid.y < hipMid.y &&
                  Math.abs(shoulderMid.x - hipMid.x) < 0.4 * torsoLen;
    var kneesStraight = jointAngle(k[11], k[13], k[15]) > 160 &&
                        jointAngle(k[12], k[14], k[16]) > 160;
    var legLen = (Math.hypot(hipMid.x-k[15].x, hipMid.y-k[15].y) +
                  Math.hypot(hipMid.x-k[16].x, hipMid.y-k[16].y)) / 2 || 1;
    var feetGrounded = Math.abs(k[15].y - k[16].y) < 0.15 * legLen;
    var armsDown = k[9].y > hipMid.y - 0.15 * torsoLen &&
                   k[10].y > hipMid.y - 0.15 * torsoLen;
    return upright && kneesStraight && feetGrounded && armsDown;
  }

  var formHistory = [], formIdx = -1;

  function classify(keypoints){
    var input = new Float32Array(51);
    for(var i = 0; i < 17; i++){
      input[i*3]   = keypoints[i].x;
      input[i*3+1] = keypoints[i].y;
      input[i*3+2] = keypoints[i].score;
    }
    var emb = embedVec(input);
    var probs = classifyEmb(emb);

    probHistory.push(probs);
    if(probHistory.length > SMOOTH_N) probHistory.shift();
    var avg = new Array(LABELS.length).fill(0);
    probHistory.forEach(function(p){
      for(var j = 0; j < avg.length; j++) avg[j] += p[j] / probHistory.length;
    });
    var order = avg.map(function(p, i){ return i; }).sort(function(a, b){ return avg[b] - avg[a]; });
    var top = order[0];

    if(top !== formIdx){ formHistory = []; formIdx = top; }
    formHistory.push(formScore(emb, top));
    if(formHistory.length > SMOOTH_N) formHistory.shift();
    var form = formHistory.reduce(function(a, b){ return a + b; }, 0) / formHistory.length;

    dbg.top3 = order.slice(0, 3).map(function(i){
      return LABELS[i].en + ' ' + (avg[i] * 100).toFixed(0) + '%';
    }).join(' · ') + ' · form ' + form.toFixed(2);
    return {idx: top, conf: avg[top], form: form};
  }

  function updateResult(res, hint){
    chips.forEach(function(c){ c.classList.remove('active'); });
    if(res && res.conf >= SHOW_CONF){
      nameEl.textContent = LABELS[res.idx].en;
      subEl.textContent = LABELS[res.idx].sa;
      confFill.style.width = Math.round(res.conf * 100) + '%';
      confLbl.textContent = Math.round(res.conf * 100) + '% confidence · form: ' + formBand(res.form);
      chips[res.idx].classList.add('active');
    }else if(res){
      nameEl.textContent = '—';
      subEl.textContent = 'maybe ' + LABELS[res.idx].en + ' (' + Math.round(res.conf * 100) + '%), hold the pose steady';
      confFill.style.width = Math.round(res.conf * 100) + '%';
      confLbl.textContent = '';
    }else{
      nameEl.textContent = '—';
      subEl.textContent = hint || 'make sure your whole body is in frame';
      confFill.style.width = '0';
      confLbl.textContent = '';
    }
  }

  function draw(keypoints){
    var w = canvas.width, h = canvas.height;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);            // selfie mirror; detection ran on the raw frame
    ctx.drawImage(video, 0, 0, w, h);
    if(keypoints){
      ctx.strokeStyle = accent();
      ctx.lineWidth = 2;
      EDGES.forEach(function(e){
        var a = keypoints[e[0]], b = keypoints[e[1]];
        if(a.score > DRAW_SCORE && b.score > DRAW_SCORE){
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      });
      ctx.fillStyle = accent();
      keypoints.forEach(function(kp){
        if(kp.score > DRAW_SCORE){
          ctx.beginPath(); ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2); ctx.fill();
        }
      });
    }
    ctx.restore();               // HUD drawn unmirrored so text is readable
    if(debugHud){
      ctx.fillStyle = 'rgba(0,0,0,.65)';
      ctx.fillRect(0, h - 58, w, 58);
      ctx.fillStyle = '#fff';
      ctx.font = '12px monospace';
      ctx.fillText('min kp score: ' + dbg.minAll.toFixed(2) + '   gate: ' + dbg.gate, 10, h - 38);
      ctx.fillText('top3: ' + (dbg.top3 || '(not classifying)'), 10, h - 18);
    }
  }

  async function loop(now){
    if(!running) return;
    var dt = now - lastT; lastT = now;
    fpsAvg = fpsAvg * 0.9 + (1000 / Math.max(dt, 1)) * 0.1;
    fpsEl.textContent = Math.round(fpsAvg) + ' fps';

    var poses = [];
    try{ poses = await detector.estimatePoses(video); }catch(e){ /* skip frame */ }
    var kps = poses.length ? poses[0].keypoints : null;

    if(weights && kps){
      dbg.minAll = groupMin(kps, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
      var torsoOk = groupMin(kps, TORSO) >= TORSO_MIN;
      var legsOk  = groupMin(kps, LEGS)  >= LEGS_MIN;
      if(torsoOk && legsOk && isNeutralStand(kps)){
        dbg.gate = 'neutral stand';
        dbg.top3 = '';
        probHistory = [];
    formHistory = [];
        updateResult(null, 'standing detected; strike one of the 11 asanas below');
      }else if(torsoOk && legsOk){
        dbg.gate = 'open';
        updateResult(classify(kps));
      }else{
        dbg.gate = torsoOk ? 'legs low' : 'torso low';
        dbg.top3 = '';
        probHistory = [];
    formHistory = [];
        updateResult(null, torsoOk
          ? 'legs not visible; step back so knees and ankles are in frame'
          : 'upper body not visible; face the camera and step back');
      }
    }else if(kps && !weights){
      dbg.gate = 'no classifier';
    }
    draw(kps);
    rafId = requestAnimationFrame(loop);
  }

  // debug hook, pairs with the "d" HUD
  window.__yogaDemo = {
    setWeights: function(w){ weights = w; },
    classifyVec: classifyVec,
    embedVec: embedVec,
    formScore: formScore
  };

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
  document.addEventListener('keydown', function(e){
    if(e.key === 'd' && !e.metaKey && !e.ctrlKey && !e.altKey &&
       !/^(input|textarea)$/i.test(document.activeElement.tagName)){
      debugHud = !debugHud;
    }
  });
  document.addEventListener('visibilitychange', function(){
    if(document.hidden && running) stop();
  });
})();
