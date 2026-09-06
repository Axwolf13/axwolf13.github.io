(function(){
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // hello, DevTools person
  try{
    console.log('%ckp_dev · 0.99 · hello, fellow engineer.', 'font-family:monospace;font-size:13px;color:#11A05A');
    console.log('Source: https://github.com/Axwolf13/axwolf13.github.io\nSay hi: akshay57ax@gmail.com\nTips: press "t" to toggle the theme. Press "v" to run object detection on this page.');
  }catch(e){}

  // theme toggle (boot script in <head> sets the initial data-theme)
  function applyTheme(t){
    document.documentElement.setAttribute('data-theme', t);
    var meta = document.querySelector('meta[name="theme-color"]');
    if(meta) meta.setAttribute('content', t === 'dark' ? '#0F1626' : '#FAFAF6');
  }
  function flipTheme(){
    var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try{ localStorage.setItem('theme', next); }catch(e){}
  }
  var toggle = document.querySelector('.theme-toggle');
  if(toggle) toggle.addEventListener('click', flipTheme);

  // vision mode: draw detection boxes over the page's structure
  var cvMap = [
    ['h1', 'headline'],
    ['.lede', 'text_block'],
    ['.badge', 'status_chip'],
    ['.pose', 'pose_input'],
    ['.pub', 'publication_card'],
    ['.proj', 'work_item'],
    ['.tl-item', 'timeline_node'],
    ['.pill-row', 'skill_cluster'],
    ['.cta-row', 'action_row'],
    ['.foot-links', 'action_row'],
    ['.sec-head', 'section_label']
  ];
  function visionOn(){
    cvMap.forEach(function(pair){
      document.querySelectorAll(pair[0]).forEach(function(el){
        var conf = (0.62 + Math.random() * 0.37).toFixed(2);
        el.setAttribute('data-cv', pair[1] + ' · ' + conf);
      });
    });
    document.body.classList.add('vision');
  }
  addEventListener('keydown', function(e){
    if(e.ctrlKey || e.metaKey || e.altKey) return;
    if(e.key === 't') flipTheme();
    if(e.key === 'v'){
      if(document.body.classList.contains('vision')){
        document.body.classList.remove('vision');
      }else{
        visionOn();
      }
    }
  });
  var mq = matchMedia('(prefers-color-scheme: dark)');
  if(mq.addEventListener){
    mq.addEventListener('change', function(e){
      var stored = null;
      try{ stored = localStorage.getItem('theme'); }catch(err){}
      if(!stored) applyTheme(e.matches ? 'dark' : 'light');
    });
  }

  // scroll reveal, with stagger for list items
  if(!reduce && 'IntersectionObserver' in window){
    var els = document.querySelectorAll('section, footer');
    els.forEach(function(el){ el.classList.add('reveal'); });
    var items = document.querySelectorAll('.proj, .tl-item');
    items.forEach(function(el){
      el.classList.add('reveal');
      var idx = Array.prototype.indexOf.call(el.parentElement.children, el);
      el.style.transitionDelay = Math.min(idx * 80, 400) + 'ms';
    });
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, {threshold:.08});
    els.forEach(function(el){ io.observe(el); });
    items.forEach(function(el){ io.observe(el); });
  }

  // active nav highlight
  if('IntersectionObserver' in window){
    var links = document.querySelectorAll('nav a[href^="#"]');
    var map = {};
    links.forEach(function(a){ map[a.getAttribute('href').slice(1)] = a; });
    var nio = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        var a = map[e.target.id];
        if(a && e.isIntersecting){
          links.forEach(function(l){ l.classList.remove('active'); });
          a.classList.add('active');
        }
      });
    }, {rootMargin:'-25% 0px -60% 0px'});
    Object.keys(map).forEach(function(id){
      var t = document.getElementById(id);
      if(t) nio.observe(t);
    });
  }

  // pose replay
  function armPose(svg){
    svg.style.cursor = 'pointer';
    svg.addEventListener('click', function(){
      var copy = svg.cloneNode(true);
      svg.replaceWith(copy);
      armPose(copy);
    });
  }
  var pose = document.querySelector('.pose');
  if(pose && !reduce) armPose(pose);

  // scroll progress bar + header shadow
  var bar = document.getElementById('progress');
  var head = document.querySelector('header');
  var onScroll = function(){
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    if(bar) bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
    if(head) head.classList.toggle('scrolled', h.scrollTop > 8);
  };
  addEventListener('scroll', onScroll, {passive:true});
  onScroll();

  // keypoint confidence jitter
  if(!reduce){
    setInterval(function(){
      document.querySelectorAll('.pose .kp-label').forEach(function(el){
        var base = el.textContent.split('·')[0].trim();
        el.textContent = base + ' · 0.' + (93 + Math.floor(Math.random() * 6));
      });
    }, 1800);
  }

  // stat count-up
  function countUp(el){
    var m = el.textContent.trim().match(/^([\d,]+)(%?)$/);
    if(!m) return;
    var target = parseInt(m[1].replace(/,/g,''), 10);
    var suffix = m[2];
    var t0 = null;
    function step(t){
      if(!t0) t0 = t;
      var p = Math.min((t - t0) / 900, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased).toLocaleString('en-US') + suffix;
      if(p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  var statsBox = document.querySelector('.stats');
  if(statsBox && !reduce && 'IntersectionObserver' in window){
    var sio = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        if(e.isIntersecting){
          e.target.querySelectorAll('.stat .n').forEach(countUp);
          sio.unobserve(e.target);
        }
      });
    }, {threshold:.4});
    sio.observe(statsBox);
  }
})();
