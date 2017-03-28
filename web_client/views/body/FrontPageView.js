import { cancelRestRequests, apiRoot, staticRoot } from 'girder/rest';
import { getCurrentUser } from 'girder/auth';
import * as version from 'girder/version';
import GirderFrontPageView from 'girder/views/body/FrontPageView';

import frontPageTemplate from '../../templates/body/frontPage.pug';
import '../../stylesheets/body/frontPage.styl';

var FrontPageView = GirderFrontPageView.extend({
    events: {},

    initialize: function () {
        cancelRestRequests('fetch');
        this.render();
    },

    render: function () {
        this.$el.html(frontPageTemplate({
            apiRoot,
            staticRoot,
            version,
            currentUser: getCurrentUser()
        }));
    }
});

export default FrontPageView;
