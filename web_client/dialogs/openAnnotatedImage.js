import { restRequest } from 'girder/rest';
import ItemCollection from 'girder/collections/ItemCollection';
import View from 'girder/views/View';
import 'girder/utilities/jquery/girderModal';

import events from '../events';
import router from '../router';

import template from '../templates/dialogs/openAnnotatedImage.pug';
import '../stylesheets/dialogs/openAnnotatedImage.styl';

let dialog;

const OpenAnnotatedImage = View.extend({
    events: {
        'click .h-annotated-image': '_submit'
    },

    initialize() {
        this.collection = new ItemCollection();
        this.listenTo(this.collection, 'reset', this.render);
    },

    render() {
        this.$el.html(template({
            items: this.collection.toJSON()
        })).girderModal(this);
        return this;
    },

    fetch() {
        return restRequest({
            path: 'annotation/images',
            data: {
                limit: 10
            }
        }).done((items) => {
            this.collection.reset(items);
        });
    },

    _submit(evt) {
        const id = this.$(evt.currentTarget).data('id');
        router.setQuery('bounds', null, {trigger: false});
        router.setQuery('image', id, {trigger: true});
        this.$el.modal('hide');
    }
});

function createDialog() {
    return new OpenAnnotatedImage({
        parentView: null
    });
}

events.on('h:openAnnotatedImageUi', function () {
    if (!dialog) {
        dialog = createDialog();
    }
    dialog.setElement($('#g-dialog-container')).fetch();
});

export default createDialog;
