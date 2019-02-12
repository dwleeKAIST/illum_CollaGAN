#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cherrypy
import random
import string
import preproc_api
import db

userpassdict  = db.get_users()
get_ha1       = cherrypy.lib.auth_digest.get_ha1_dict(userpassdict)


host = sys.argv[1]
port = int(sys.argv[2])

config = {
  'global' : {
        'server.socket_host': host,
        'server.socket_port': port,

        #'server.ssl_module':'pyopenssl',
        #'server.ssl_certificate':"/home/wbim/.ssl/server.crt",
        #'server.ssl_private_key':"/home/wbim/.ssl/server.key",
  },
  '/' : {
    # HTTP verb dispatcher
    'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
    # JSON response
    # 'tools.json_out.on' : True,
    # Digest Auth
    'tools.auth_digest.on'      : False,
    'tools.auth_digest.realm'   : db.auth_realm,
    'tools.auth_digest.get_ha1' : get_ha1,
    'tools.auth_digest.key'     : ''.join([random.choice(string.ascii_letters) for _ in range(64)]),
  }
}

class BISPL_API(object):
    def __init__(self):
        print('init BISPL_API')
        #self.denoise_core = preproc_api.DenoiseCore()
        #self.SISR_core    = preproc_api.SISRCore()
        self.illum_core = preproc_api.IllumCore()
        self.switcher = {'illum':self.illum_core}
        #'denoise': self.denoise_core,'SISR': self.SISR_core}

    def _cp_dispatch(self, vpath):
        if len(vpath) > 0 and vpath.pop(0) == 'api-BISPL':
            if len(vpath) > 0:
                return self.switcher[vpath.pop(0)]
        return vpath


if __name__ == '__main__':
    cherrypy.quickstart(BISPL_API(), '/', config)
    
